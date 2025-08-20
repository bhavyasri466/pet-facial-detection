import cv2
import pickle
import numpy as np
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
import gridfs
import face_recognition
import base64
from io import BytesIO
from PIL import Image
import logging

# Configure logging to match command-line output style
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FrameProcessor:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        # Validate config
        required_keys = ['mongo_uri', 'db_name', 'pet_date', 'batch']
        missing = [key for key in required_keys if key not in config]
        if missing:
            raise ValueError(f"Missing config keys: {missing}")
        
        try:
            self.client = MongoClient(config['mongo_uri'])
            self.db = self.client[config['db_name']]
            self.fs = gridfs.GridFS(self.db)
            self.fs_wrong = gridfs.GridFS(self.db, 'wrong_tag_images')
            self.detected_collection = self.db['detected_frames']
            self.matched_collection = self.db['matched_results']
            self.wrong_tag_collection = self.db['wrong_tag_images']
            self.logger.info(f"Connected to MongoDB database: {config['db_name']}")
            
            # Create indexes for performance
            self.detected_collection.create_index([('image_id', 1)])
            self.matched_collection.create_index([('bib_number', 1), ('timestamp', 1)])
            self.wrong_tag_collection.create_index([('image', 1)])
            
            # Initialize storage for precomputed encodings
            self.known_encodings = []
            self.known_metadata = []
        except Exception as e:
            self.logger.error(f"Failed to initialize MongoDB connection: {str(e)}")
            raise

    def load_pickle(self, pickle_path):
        self.logger.info(f"Loading pickle file: {pickle_path}")
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            self.logger.info(f"Pickle file structure: {type(data)}")
            if isinstance(data, list) and len(data) > 0:
                self.logger.info(f"First entry keys: {data[0].keys()}")
                self.logger.info(f"Image type: {type(data[0]['image'])}")
            
            # Precompute face encodings
            for item in data:
                try:
                    if not all(key in item for key in ['image', 'bib', 'roll_no']):
                        self.logger.warning(f"Skipping entry due to missing keys: {item.keys()}")
                        continue
                    image_data = item['image']
                    if isinstance(image_data, Image.Image):
                        image = np.array(image_data)
                        if image.ndim == 2:
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    elif isinstance(image_data, str):
                        image_data = base64.b64decode(image_data)
                        image_array = np.frombuffer(image_data, np.uint8)
                        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    elif isinstance(image_data, np.ndarray):
                        image = image_data
                        if image.ndim == 2:
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    else:
                        self.logger.warning(f"Unsupported image type for bib {item.get('bib', 'unknown')}: {type(image_data)}")
                        continue
                    
                    rgb_image = image if image.shape[-1] == 3 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_image)
                    if encodings:
                        self.known_encodings.append(encodings[0])
                        self.known_metadata.append({
                            'bib': item['bib'],
                            'roll_no': item['roll_no']
                        })
                        self.logger.info(f"Encoded face for bib {item.get('bib')}")
                    else:
                        self.logger.warning(f"No faces found in pickle image for bib {item.get('bib', 'unknown')}")
                except Exception as e:
                    self.logger.error(f"Error encoding pickle image for bib {item.get('bib', 'unknown')}: {str(e)}")
            self.logger.info(f"Successfully loaded and encoded {len(self.known_encodings)} faces from pickle file")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load pickle file {pickle_path}: {str(e)}")
            raise

    def img_to_base64_cv2(self, image):
        try:
            retval, buffer = cv2.imencode('.jpg', image)
            if not retval:
                raise Exception("Failed to encode image to base64")
            jpg_as_text = base64.b64encode(buffer)
            return jpg_as_text
        except Exception as e:
            self.logger.error(f"Error converting image to base64: {str(e)}")
            raise

    def save_frame_to_gridfs(self, frame):
        try:
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                raise Exception("Failed to encode frame")
            image_bytes = buffer.tobytes()
            image_id = self.fs.put(image_bytes, content_type='image/jpeg')
            self.logger.debug(f"Saved frame to GridFS with ID: {image_id}")
            return image_id
        except Exception as e:
            self.logger.error(f"Error saving frame to GridFS: {str(e)}")
            raise

    def insert_wrong(self, bib_number, timestamp, pet_date, image):
        try:
            img_string = self.img_to_base64_cv2(image)
            img_id = self.fs_wrong.put(img_string)
            mydict = {
                "bib_number": bib_number,
                "timestamp": timestamp,
                "pet_date": pet_date,
                "fetch_flag": None,
                "image": img_id
            }
            self.wrong_tag_collection.insert_one(mydict)
            self.logger.debug(f"Inserted wrong tag image with ID: {img_id}")
        except Exception as e:
            self.logger.error(f"Error inserting wrong tag image: {str(e)}")
            raise

    def compare_faces(self, frame, frame_count):
        try:
            # Resize frame for faster processing (optional)
            frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_encodings = face_recognition.face_encodings(rgb_frame)
            frame_locations = face_recognition.face_locations(rgb_frame)
            self.logger.info(f"Frame {frame_count}: {len(frame_encodings)} faces detected")
            
            matches = []
            for frame_encoding, frame_location in zip(frame_encodings, frame_locations):
                for known_encoding, metadata in zip(self.known_encodings, self.known_metadata):
                    result = face_recognition.compare_faces([known_encoding], frame_encoding)
                    confidence = face_recognition.face_distance([known_encoding], frame_encoding)[0]
                    if result[0]:
                        matches.append({
                            'bib': metadata['bib'],
                            'roll_no': metadata['roll_no'],
                            'confidence': 1 - confidence,
                            'box': frame_location,
                            'method': 'face_recognition'
                        })
                        self.logger.debug(f"Match found for bib {metadata['bib']} with confidence {1 - confidence}")
            return matches, len(frame_encodings) > 0
        except Exception as e:
            self.logger.error(f"Face recognition error in frame {frame_count}: {str(e)}")
            return [], False

    def process_video(self, video_path, pickle_path):
        self.logger.info(f"Starting frame processing for video: {video_path}")
        try:
            # Load pickle data
            pickle_data = self.load_pickle(pickle_path)
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Error opening video file: {video_path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.logger.info(f"Opened video: {video_path}, FPS: {fps}")
            
            frame_interval = int(fps / 3)  # Every 3rd frame per second
            frame_count = 0
            saved_frame_count = 0
            detected_docs = []
            matched_docs = []
            batch_size = 100
            
            while cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.info(f"End of video at frame {frame_count}")
                        break
                except Exception as e:
                    self.logger.error(f"Error reading frame {frame_count}: {str(e)}")
                    break
                
                frame_count += 1
                if frame_count % frame_interval != 0:
                    continue
                
                timestamp = frame_count / fps
                
                # Save frame to GridFS
                try:
                    image_id = self.save_frame_to_gridfs(frame)
                except Exception as e:
                    self.logger.error(f"Error saving frame {frame_count}: {str(e)}")
                    continue
                
                # Save detected frame metadata
                detected_doc = {
                    'frame_count': frame_count,
                    'timestamp': timestamp,
                    'image_id': image_id,
                    'pet_date': self.config['pet_date'],
                    'batch_no': self.config['batch'],
                    'processing_time': datetime.utcnow()
                }
                detected_docs.append(detected_doc)
                
                # Compare with pickle data
                matches, has_faces = self.compare_faces(frame, frame_count)
                
                # If no faces detected, insert into wrong_tag_images
                if not has_faces:
                    try:
                        self.insert_wrong(
                            bib_number="unknown",
                            timestamp=timestamp,
                            pet_date=self.config['pet_date'],
                            image=frame
                        )
                    except Exception as e:
                        self.logger.error(f"Error inserting wrong tag for frame {frame_count}: {str(e)}")
                
                # Save matched results
                for match in matches:
                    doc = {
                        'bib_number': str(match['bib']),
                        'rollno': str(match['roll_no']),
                        'pet_date': str(self.config['pet_date']),
                        'batch_no': int(self.config['batch']),
                        'timestamp': timestamp,
                        'frame_count': int(frame_count),
                        'confidence': float(match['confidence']),
                        'face_location': [int(x) for x in match['box']],
                        'detection_method': str(match['method']),
                        'image': image_id,
                        'ai_unique_id': str(ObjectId()),
                        'processing_time': datetime.utcnow()
                    }
                    matched_docs.append(doc)
                
                # Batch insert to MongoDB
                if len(detected_docs) >= batch_size:
                    try:
                        self.detected_collection.insert_many(detected_docs)
                        self.logger.debug(f"Inserted {len(detected_docs)} detected frames")
                        detected_docs = []
                    except Exception as e:
                        self.logger.error(f"Error inserting detected frames: {str(e)}")
                    if matched_docs:
                        try:
                            self.matched_collection.insert_many(matched_docs)
                            self.logger.debug(f"Inserted {len(matched_docs)} matched results")
                            matched_docs = []
                        except Exception as e:
                            self.logger.error(f"Error inserting matched results: {str(e)}")
                
                saved_frame_count += 1
                self.logger.info(f"Processed frame {frame_count} at {timestamp:.2f} seconds")
            
            # Insert remaining documents
            if detected_docs:
                try:
                    self.detected_collection.insert_many(detected_docs)
                    self.logger.debug(f"Inserted {len(detected_docs)} detected frames")
                except Exception as e:
                    self.logger.error(f"Error inserting detected frames: {str(e)}")
            if matched_docs:
                try:
                    self.matched_collection.insert_many(matched_docs)
                    self.logger.debug(f"Inserted {len(matched_docs)} matched results")
                except Exception as e:
                    self.logger.error(f"Error inserting matched results: {str(e)}")
            
            cap.release()
            self.logger.info(f"Total frames processed: {saved_frame_count}")
        except Exception as e:
            self.logger.error(f"Error processing video {video_path}: {str(e)}")
            if 'cap' in locals():
                cap.release()
            raise

    def close(self):
        try:
            self.client.close()
            self.logger.info("Closed MongoDB connection")
        except Exception as e:
            self.logger.error(f"Error closing MongoDB connection: {str(e)}")

# Usage example
if __name__ == "__main__":
    config = {
        'mongo_uri': 'mongodb://localhost:27017/',
        'db_name': 'PATNA_2025-06-24',
        'pet_date': '2025-06-24',
        'batch': 1
    }
    
    processor = FrameProcessor(config)
    try:
        processor.process_video('D02_20250503061702.mp4', 'images_2025-05-03_batch1.pickle')
    except Exception as e:
        processor.logger.error(f"Script execution failed: {str(e)}")
    finally:
        processor.close()