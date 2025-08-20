import cv2
import logging
import numpy as np
from bson.objectid import ObjectId
from datetime import datetime
import os
import pickle
import face_recognition
try:
    from pymongo import MongoClient
    from gridfs import GridFS
except ImportError as e:
    print(f"Error: Required MongoDB modules not found. Install with: `pip install pymongo`")
    exit(1)
try:
    import pytesseract
except ImportError:
    print("Warning: pytesseract not installed. Bib detection disabled. Install with: `pip install pytesseract`")
    pytesseract = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('framescutinsertion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# MongoDB connection
def connect_to_mongodb(db_name):
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client[db_name]
        collection = db["frames"]
        gridfs = GridFS(db)
        logger.info(f"Connected to MongoDB database: {db_name}")
        return collection, gridfs
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        exit(1)

# Load pickle file
def load_pickle_data(pickle_path):
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded pickle file: {pickle_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading pickle file {pickle_path}: {str(e)}")
        return None

# Bib detection using OCR
def detect_bib(image):
    try:
        if not pytesseract:
            logger.warning("pytesseract not available, skipping bib detection")
            return None, False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        text = pytesseract.image_to_string(thresh, config='--psm 6 digits')
        bib_number = text.strip() if text.strip().isdigit() else None
        bib_detected = bib_number is not None
        logger.info(f"Bib detection: Number={bib_number}, Detected={bib_detected}")
        return bib_number, bib_detected
    except Exception as e:
        logger.error(f"Bib detection error: {str(e)}")
        return None, False

# Face detection and recognition
def process_frame(frame, frame_id, pet_date, batch, pickle_data, collection):
    try:
        # Validate frame
        if frame is None or frame.shape[0] == 0:
            logger.error(f"Invalid frame for frame_id: {frame_id}")
            return {
                "rollno": None,
                "bib_number": None,
                "frame_id": frame_id,
                "pet_date": pet_date,
                "batch": batch,
                "bib_detection_flag": False,
                "final_verified_bib": None,
                "output": "Invalid frame",
                "timestamp": datetime.utcnow()
            }
        
        # Convert to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Match faces against pickle data
        rollno = None
        if pickle_data and face_encodings:
            for encoding in face_encodings:
                for ref_encoding, ref_bib in pickle_data:
                    match = face_recognition.compare_faces([ref_encoding], encoding, tolerance=0.5)
                    if match[0]:
                        rollno = str(ref_bib)
                        break
                if rollno:
                    break
        
        # Detect bib
        bib_number, bib_detected = detect_bib(frame)
        
        # Prepare document
        output = f"Detected {len(face_locations)} faces" if face_locations else "No faces detected"
        document = {
            "rollno": rollno,
            "bib_number": bib_number,
            "frame_id": frame_id,
            "pet_date": pet_date,
            "batch": batch,
            "bib_detection_flag": bib_detected,
            "final_verified_bib": None,
            "output": output,
            "timestamp": datetime.utcnow()
        }
        
        # Log results
        logger.info(f"Frame ID: {frame_id}, Faces: {len(face_locations)}, Rollno: {rollno}, Bib: {bib_number}, Bib Detected: {bib_detected}, Output: {output}")
        
        # Save debug image
        if not face_locations:
            debug_path = f"debug_no_faces_{frame_id}.jpg"
            cv2.imwrite(debug_path, frame)
            logger.info(f"Saved debug image: {debug_path}")
        
        return document
    
    except Exception as e:
        logger.error(f"Error processing frame_id {frame_id}: {str(e)}")
        return {
            "rollno": None,
            "bib_number": None,
            "frame_id": frame_id,
            "pet_date": pet_date,
            "batch": batch,
            "bib_detection_flag": False,
            "final_verified_bib": None,
            "output": f"Error: {str(e)}",
            "timestamp": datetime.utcnow()
        }

def process_video(video_path, pet_date, batch, pickle_data, collection, gridfs):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame to GridFS
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logger.error(f"Failed to encode frame {frame_count}")
                continue
            frame_id = gridfs.put(buffer.tobytes(), filename=f"frame_{frame_count}.jpg")
            
            # Process frame
            document = process_frame(frame, frame_id, pet_date, batch, pickle_data, collection)
            
            # Insert into MongoDB
            collection.insert_one(document)
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Processed {frame_count} frames from video: {video_path}")
    
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}")

def main():
    # Hardcoded configuration
    db_config = {'database_name': 'PATNA_2025-05-03'}
    video_path = 'D02_20250503061702.mp4'
    pickle_path = 'images_2025-05-03_batch1.pickle'
    pet_date = '2025-05-03'  # Corrected to match database and video
    batch = '1'
    
    # Connect to MongoDB
    collection, gridfs = connect_to_mongodb(db_config['database_name'])
    
    # Load pickle data
    pickle_data = load_pickle_data(pickle_path)
    
    # Process video
    logger.info("Starting frame processing")
    process_video(video_path, pet_date, batch, pickle_data, collection, gridfs)
    logger.info("Processing complete")

if __name__ == "__main__":
    main()