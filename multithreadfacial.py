import os
import cv2
import time
import uuid
import base64
import pickle
import logging
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
from pymongo import MongoClient
import gridfs
import face_recognition
from bson.objectid import ObjectId
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_frame_external(frame, known_encodings, known_metadata, encoding_threshold, pet_date, db_config):
    try:
        client = MongoClient(db_config['mongo_uri'])
        db = client[str(db_config['database_name'])]  # Ensure database_name is string
        wrong_tag_fs = gridfs.GridFS(db, collection='wrong_tag_images')

        img_data = wrong_tag_fs.get(frame['image']).read()
        try:
            img = np.frombuffer(base64.b64decode(img_data), np.uint8)
        except:
            img = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if image is None:
            db['wrong_tag_results'].update_one({"_id": frame["_id"]}, {"$set": {"facial_flag": 1}})
            return None

        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        if not face_encodings:
            db['wrong_tag_results'].update_one({"_id": frame["_id"]}, {"$set": {"facial_flag": 1}})
            return None

        for encoding in face_encodings:
            distances = face_recognition.face_distance(known_encodings, encoding)
            if len(distances) == 0:
                continue
            best_idx = np.argmin(distances)
            best_distance = distances[best_idx]
            if best_distance < encoding_threshold:
                match_result = known_metadata[best_idx]
                bib_padded = str(match_result['bib']).zfill(4)

                record = {
                    "bib_number": bib_padded,
                    "rollno": str(match_result['roll_no']),
                    "pet_date": pet_date,
                    "timestamp": frame.get("timestamp"),
                    "operator_timestamp": frame.get("timestamp"),
                    "facial_flag": "1",
                    "output": "F",
                    "ai_unique_id": str(frame.get("_id", uuid.uuid4())),
                    "image": frame['image'],
                    "confidence": float(1 - best_distance),
                    "detection_method": "face_recognition",
                    "frame_processed_at": datetime.now()
                }
                db['bib_detection_results'].insert_one(record)
                break

        db['wrong_tag_results'].update_one({"_id": frame["_id"]}, {"$set": {"facial_flag": 1}})
        return str(frame['_id'])
    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return None

class WrongTagFacialProcessor:
    def __init__(self, config_path='mongodb_config.properties', pickle_folder=None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.db_name = str(self.config.get('database_name'))  # Ensure string
        self.pet_date = self.config.get('petdate', datetime.now().strftime('%d-%m-%Y'))
        self.mongo_uri = self.config.get('mongo_uri', 'mongodb://localhost:27017/')
        self.encoding_threshold = float(self.config.get('encoding_threshold', 0.6))

        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]

        self.known_encodings = []
        self.known_metadata = []

        self.pickle_folder = pickle_folder
        self.loaded_pickle_files = set()
        self.excel_logs = []

        self.excel_filename = f"facial_logs_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.xlsx"

        if self.pickle_folder:
            self.load_all_pickles_from_folder()

    def _load_config(self, filepath):
        config = {}
        if os.path.exists(filepath):
            with open(filepath) as f:
                for line in f:
                    if '=' in line:
                        k, v = line.strip().split('=', 1)
                        config[k.strip()] = v.strip()
        return config

    def load_all_pickles_from_folder(self):
        if not os.path.exists(self.pickle_folder):
            self.logger.warning(f"Pickle folder '{self.pickle_folder}' does not exist.")
            return

        for filename in os.listdir(self.pickle_folder):
            if filename.endswith(".pickle") and filename not in self.loaded_pickle_files:
                full_path = os.path.join(self.pickle_folder, filename)
                self.load_pickle(full_path)
                self.loaded_pickle_files.add(filename)

    def load_pickle(self, pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            self.logger.info(f"Loaded {len(data)} registration records from '{os.path.basename(pickle_path)}'")

            for item in data:
                try:
                    img_data = item.get('image')
                    if isinstance(img_data, Image.Image):
                        image = np.array(img_data)
                    elif isinstance(img_data, str):
                        image = cv2.imdecode(np.frombuffer(base64.b64decode(img_data), np.uint8), cv2.IMREAD_COLOR)
                    elif isinstance(img_data, np.ndarray):
                        image = img_data
                    else:
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
                except Exception as e:
                    self.logger.error(f"Failed to process registration image in {os.path.basename(pickle_path)}: {e}")
        except Exception as e:
            self.logger.error(f"Error loading pickle file {pickle_path}: {e}")

    def get_unprocessed_wrong_tag_frames(self):
        return list(self.db['wrong_tag_results'].find({
            "facial_flag": {"$ne": 1},
            "image": {"$exists": True},
            "timestamp": {"$exists": True, "$type": "number"}
        }).sort("timestamp", 1))

    def process_wrong_tag_frames_parallel(self):
        frames = self.get_unprocessed_wrong_tag_frames()
        self.logger.info(f"Found {len(frames)} unprocessed wrong_tag frames (sorted by timestamp).")

        db_config = {
            'database_name': self.db_name,
            'mongo_uri': self.mongo_uri
        }

        args = [
            (frame, self.known_encodings, self.known_metadata, self.encoding_threshold, self.pet_date, db_config)
            for frame in frames
        ]

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(process_frame_external, *arg) for arg in args]
            for future in as_completed(futures):
                future.result()

    def save_logs_to_excel(self, file_path=None):
        if not self.excel_logs:
            return

        if file_path is None:
            file_path = self.excel_filename

        df = pd.DataFrame(self.excel_logs)
        try:
            df.to_excel(file_path, index=False)
            self.logger.info(f"Excel log saved to: {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving Excel log: {e}")

    def close(self):
        self.client.close()

if __name__ == "__main__":
    processor = WrongTagFacialProcessor(
        config_path='mongodb_config.properties',
        pickle_folder='singlebatch'
    )
    try:
        processor.process_wrong_tag_frames_parallel()
    except Exception as e:
        logging.error(f"Fatal error during processing: {e}", exc_info=True)
    finally:
        processor.save_logs_to_excel()
        processor.close()
