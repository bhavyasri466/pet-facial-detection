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
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WrongTagFacialProcessor:
    def __init__(self, config_path='mongodb_config.properties', pickle_folder=None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.db_name = self.config.get('database_name')
        self.pet_date = self.config.get('petdate', datetime.now().strftime('%d-%m-%Y'))
        self.mongo_uri = self.config.get('mongo_uri', 'mongodb://localhost:27017/')
        self.encoding_threshold = float(self.config.get('encoding_threshold', 0.6))

        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.wrong_tag_fs = gridfs.GridFS(self.db, collection='wrong_tag_images')

        self.known_encodings = []
        self.known_metadata = []
        self.pickle_folder = pickle_folder
        self.loaded_pickle_files = set()
        self.excel_logs = []
        self.lock = Lock()

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
                    encodings = face_recognition.face_encodings(image, num_jitters=1, model='cnn')
                    if encodings:
                        self.known_encodings.append(encodings[0])
                        self.known_metadata.append({
                            'bib': item['bib'],
                            'roll_no': item['roll_no'],
                            'image_data': img_data
                        })
                    else:
                        self.logger.warning(f"No face found for bib {item.get('bib')} in {os.path.basename(pickle_path)}")
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

    def get_pet_image(self, image_id):
        try:
            img_data = self.wrong_tag_fs.get(image_id).read()
            try:
                img = np.frombuffer(base64.b64decode(img_data), np.uint8)
            except:
                img = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            self.logger.error(f"Failed to retrieve image {image_id}: {e}")
            return None

    def process_wrong_tag_frames(self, max_workers=10):
        frames = self.get_unprocessed_wrong_tag_frames()
        self.logger.info(f"Found {len(frames)} unprocessed wrong_tag frames.")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_single_frame, frame) for frame in frames]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Thread error: {e}")

        self.save_logs_to_excel()

    def process_single_frame(self, frame):
        t0 = time.time()
        frame_fetch_start = time.time()
        image_data = self.get_pet_image(frame['image'])
        frame_fetch_time_ms = (time.time() - frame_fetch_start) * 1000

        if image_data is None:
            self.db['wrong_tag_results'].update_one({"_id": frame["_id"]}, {"$set": {"facial_flag": 1}})
            return

        frame_capture_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        face_detect_start = time.time()
        small_frame = cv2.resize(image_data, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model='cnn')

        if not face_locations:
            self.db['wrong_tag_results'].update_one({"_id": frame["_id"]}, {"$set": {"facial_flag": 1}})
            return

        # Select the largest face only
        face_locations = sorted(face_locations, key=lambda rect: (rect[2] - rect[0]) * (rect[1] - rect[3]), reverse=True)
        largest_location = face_locations[0]
        face_encodings = face_recognition.face_encodings(rgb_frame, [largest_location], num_jitters=1, model='cnn')
        face_detect_time_ms = (time.time() - face_detect_start) * 1000

        if not face_encodings:
            self.db['wrong_tag_results'].update_one({"_id": frame["_id"]}, {"$set": {"facial_flag": 1}})
            return

        matched = False
        encoding = face_encodings[0]
        compare_start = time.time()
        distances = face_recognition.face_distance(self.known_encodings, encoding)
        compare_time_ms = (time.time() - compare_start) * 1000

        if len(distances) == 0:
            self.db['wrong_tag_results'].update_one({"_id": frame["_id"]}, {"$set": {"facial_flag": 1}})
            return

        best_idx = np.argmin(distances)
        best_distance = distances[best_idx]

        if best_distance < self.encoding_threshold:
            matched = True
            match_result = self.known_metadata[best_idx]
            bib_padded = str(match_result['bib']).zfill(4)
            insert_start = time.time()

            try:
                record = {
                    "bib_number": bib_padded,
                    "rollno": str(match_result['roll_no']),
                    "pet_date": self.pet_date,
                    "batch_no": None,
                    "timestamp": frame.get("timestamp"),
                    "operator_timestamp": frame.get("timestamp"),
                    "Lap": None,
                    "bib_detection_flag": None,
                    "facial_flag": "1",
                    "output": "F",
                    "fetch_flag": None,
                    "manual_bib_fetch_flag": None,
                    "final_verified_bib": None,
                    "ai_unique_id": str(frame.get("_id", uuid.uuid4())),
                    "image": frame['image'],
                    "confidence": float(1 - best_distance),
                    "detection_method": "face_recognition",
                    "frame_capture_time": frame_capture_time,
                    "facial_comparison_time_ms": compare_time_ms,
                    "frame_processed_at": datetime.now()
                }

                self.db['bib_detection_results'].insert_one(record)
                insert_time_ms = (time.time() - insert_start) * 1000

                with self.lock:
                    self.excel_logs.append({
                        "frame_id": str(frame['_id']),
                        "bib_number": bib_padded,
                        "frame_capture_time": frame_capture_time,
                        "frame_fetch_time_ms": frame_fetch_time_ms,
                        "face_detect_time_ms": face_detect_time_ms,
                        "facial_comparison_time_ms": compare_time_ms,
                        "mongodb_insertion_time_ms": insert_time_ms,
                        "face_processing_total_time_ms": int((time.time() - t0) * 1000)
                    })

            except Exception as db_err:
                self.logger.error(f"DB insert failed for frame {frame['_id']}: {db_err}")

        self.db['wrong_tag_results'].update_one({"_id": frame["_id"]}, {"$set": {"facial_flag": 1}})

    def save_logs_to_excel(self, file_path=None):
        if not self.excel_logs:
            self.logger.info("No logs to save.")
            return

        if file_path is None:
            file_path = self.excel_filename

        df = pd.DataFrame(self.excel_logs)
        cols = [
            "frame_id", "bib_number", "frame_capture_time", "frame_fetch_time_ms",
            "face_detect_time_ms", "facial_comparison_time_ms", "mongodb_insertion_time_ms",
            "face_processing_total_time_ms"
        ]
        df = df[cols]
        df.columns = [
            "Frame ID", "Bib Number", "Frame Capture Time", "Frame Fetch Time (ms)",
            "Face Detection Time (ms)", "Facial Comparison Time (ms)", "MongoDB Insertion Time (ms)",
            "Total Processing Time (ms)"
        ]

        try:
            df.to_excel(file_path, index=False)
            self.logger.info(f"Excel log saved to: {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving Excel log: {e}")

    def close(self):
        self.client.close()

if __name__ == '__main__':
    processor = WrongTagFacialProcessor(
        config_path='mongodb_config.properties',
        pickle_folder='regimage0807'
    )
    try:
        processor.process_wrong_tag_frames(max_workers=8)
    except Exception as e:
        logging.error(f"Fatal error during processing: {e}", exc_info=True)
    finally:
        processor.save_logs_to_excel()
        processor.close()
