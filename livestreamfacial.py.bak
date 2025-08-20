# Updated version of the facial recognition stream processor with:
# - Auto-reconnect after frame failures
# - Saving no-face frames for debugging
# - Improved encoding threshold handling

import cv2
import pickle
import numpy as np
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
import gridfs
import face_recognition
import base64
import os
import time
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FrameProcessor:
    def __init__(self, config_file='mongodb_config.properties'):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_file)
        self.config.setdefault('mongo_uri', f"mongodb://{self.config.get('facial_mongodb_ip', 'localhost')}:27017/")
        self.config.setdefault('db_name', 'Patna_2025-05-03')
        self.config.setdefault('pet_date', datetime.now().strftime('%Y-%m-%d'))
        self.config.setdefault('camera_ip', self.config.get('camera_ip_2', ''))
        self.config.setdefault('opencv_backend', 'FFMPEG')
        self.config.setdefault('encoding_threshold', 0.6)

        self.client = MongoClient(self.config['mongo_uri'])
        self.db = self.client[self.config['db_name']]
        self.fs = gridfs.GridFS(self.db)
        self.matched_collection = self.db.get_collection(self.config.get('bib_recognition_tablename', 'matched_results'))
        self.matched_collection.create_index([('bib_number', 1), ('timestamp', 1)])

        self.logger.info(f"Connected to MongoDB: {self.config['db_name']}")

        self.known_encodings = []
        self.known_metadata = []
        self.cap = None

    def _load_config(self, path):
        config = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if '=' in line:
                        k, v = line.strip().split('=', 1)
                        config[k.strip()] = v.strip()
        return config

    def load_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        self.logger.info(f"Loaded {len(data)} registration records from pickle file")

        for item in data:
            try:
                if not all(k in item for k in ['image', 'bib', 'roll_no']):
                    continue

                img_data = item['image']
                if isinstance(img_data, Image.Image):
                    image = np.array(img_data)
                elif isinstance(img_data, str):
                    image = cv2.imdecode(np.frombuffer(base64.b64decode(img_data), np.uint8), cv2.IMREAD_COLOR)
                elif isinstance(img_data, np.ndarray):
                    image = img_data
                else:
                    continue

                if image is None or image.size == 0:
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.known_encodings.append(encodings[0])
                    self.known_metadata.append({
                        'bib': item['bib'],
                        'roll_no': item['roll_no'],
                        'image_data': img_data
                    })
                else:
                    self.logger.warning(f"No faces found for bib {item['bib']}")
            except Exception as e:
                self.logger.error(f"Error processing bib {item.get('bib', 'unknown')}: {e}")

        self.logger.info(f"Encoded {len(self.known_encodings)} faces")
    def _validate_frame(self, frame):
        if frame is None:
            self.logger.debug("Frame is None")
            return None
        if not isinstance(frame, np.ndarray):
            self.logger.warning(f"Frame is not a numpy array: {type(frame)}")
            return None
        if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            self.logger.warning("Invalid frame dimensions or empty frame")
            return None
        return frame

    def start_stream(self, camera_key='camera_ip_2'):
        max_retries = 50
        retry_delay = 5
        backend = getattr(cv2, f"CAP_{self.config['opencv_backend']}", cv2.CAP_FFMPEG)
    
        self.logger.info(f"Using OpenCV backend: {self.config['opencv_backend']}")
        build_info = cv2.getBuildInformation()
        if "FFmpeg" not in build_info and "GStreamer" not in build_info:
            self.logger.warning("OpenCV may lack RTSP support. Check FFmpeg or GStreamer in build info.")
    
        self.stream_source = self.config.get(camera_key)
        if not self.stream_source:
            self.logger.error(f"No camera IP found for key: {camera_key}")
            return False
    
        for attempt in range(max_retries):
            self.logger.info(f"Attempting to start stream: {self.stream_source} (attempt {attempt + 1})")
            self.cap = cv2.VideoCapture(self.stream_source, backend)
            time.sleep(2)  # give time to establish stream
    
            if self.cap.isOpened():
                for _ in range(10):  # try reading several frames to allow stream to warm up
                    ret, frame = self.cap.read()
                    if ret and self._validate_frame(frame) is not None:
                        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        fps = self.cap.get(cv2.CAP_PROP_FPS)
                        self.logger.info(f"Stream opened: {width}x{height} @ {fps} FPS")
                        return True
                    else:
                        self.logger.warning(f"Frame not ready yet (attempt {attempt + 1}). Retrying frame read...")
                        time.sleep(1)
    
            self.logger.warning(f"Stream not ready or no valid frame. Releasing and retrying...")
            self.cap.release()
            time.sleep(retry_delay)
    
        self.logger.error("Failed to open video stream after retries")
        return False


    def compare_faces(self, frame, frame_count):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        matches = []
        if not face_encodings:
            cv2.imwrite(f"noface_frame_{frame_count}.jpg", frame)

        for encoding, location in zip(face_encodings, face_locations):
            distances = face_recognition.face_distance(self.known_encodings, encoding)
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            if best_distance < float(self.config['encoding_threshold']):
                matches.append({
                    'bib': self.known_metadata[best_match_index]['bib'],
                    'roll_no': self.known_metadata[best_match_index]['roll_no'],
                    'confidence': 1 - best_distance,
                    'location': location,
                    'method': 'face_recognition'
                })
        return matches

    def process_live_stream(self, pickle_path):
        self.load_pickle(pickle_path)
        frame_count = 0
        fail_count = 0
        max_fails = 10

        while True:
            if not self.cap or not self.cap.isOpened():
                try:
                    self.logger.info("Attempting to reconnect to stream...")
                    self.start_stream()
                    fail_count = 0
                    continue
                except Exception as e:
                    self.logger.error(f"Reconnection failed: {e}")
                    time.sleep(5)
                    continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                fail_count += 1
                self.logger.warning("Failed to read frame")
                if fail_count >= max_fails:
                    self.cap.release()
                time.sleep(0.5)
                continue

            fail_count = 0
            frame_count += 1
            if frame_count % 5 != 0:
                continue

            matches = self.compare_faces(frame, frame_count)
            timestamp = datetime.utcnow()
            for match in matches:
                try:
                    _, buffer = cv2.imencode(".jpg", frame)
                    doc = {
                        'bib_number': str(match['bib']),
                        'roll_no': str(match['roll_no']),
                        'pet_date': self.config['pet_date'],
                        'timestamp': timestamp,
                        'frame_count': frame_count,
                        'confidence': float(match['confidence']),
                        'detection_method': match['method'],
                        'frame_image': base64.b64encode(buffer),
                        'processing_time': timestamp,
                        'ai_unique_id': str(ObjectId())
                    }
                    self.matched_collection.insert_one(doc)
                    self.logger.info(f"Saved match: {match['bib']} at frame {frame_count}")
                except Exception as e:
                    self.logger.error(f"DB insert error: {e}")

    def close(self):
        if self.cap:
            self.cap.release()
        if self.client:
            self.client.close()
        cv2.destroyAllWindows()
        self.logger.info("Resources released")

if __name__ == '__main__':
    processor = FrameProcessor()
    try:
        processor.start_stream()
        processor.process_live_stream('batch_1_images.pickle')
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
    finally:
        processor.close()
