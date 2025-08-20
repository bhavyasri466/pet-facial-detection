import cv2
import os
import pyodbc
import logging
import base64
import uuid
import pickle
import numpy as np
import tensorflow as tf
import mxnet as mx
from datetime import datetime
from pymongo import MongoClient
from gridfs import GridFS
from mtcnn_detector import MtcnnDetector
import facenet
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageFile
import io
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('facial_recognition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Allow PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configuration
CONFIG = {
    # SQL Server Configuration
    "sql_server": "192.168.1.100",
    "sql_db": "bsf_new_pet_2025",
    "sql_username": "SA",
    "sql_password": "MSt8f7hxdo",
    
    # MongoDB Configuration
    "mongodb_uri": "mongodb://localhost:27017/",
    "mongodb_name": "Meer_2025-05-13",
    
    # Processing Parameters
    "batch_size": 500,
    "num_workers": 8,
    "similarity_threshold": 0.7,
    
    # Collection Names
    "collections": {
        "bib_recognition": "bib_detection_results",
        "wrong_tag_results": "wrong_tag_results",
        "facial_results": "facial_recognition_results",
        "facial_detections": "facial_detection_details",
        "images": "images"
    }
}

# Global models
global_models = {
    'detector': None,
    'session': None,
    'graph': None
}

def init_models():
    """Initialize face detection and recognition models"""
    try:
        logger.info("Initializing face detection model...")
        global_models['detector'] = MtcnnDetector(
            model_folder='model',
            ctx=mx.cpu(1),
            num_worker=4,
            accurate_landmark=False
        )
        
        logger.info("Initializing face recognition model...")
        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        global_models['session'] = tf.compat.v1.Session(config=tf_config)
        
        with global_models['session'].as_default():
            with tf.compat.v1.gfile.GFile('facial_embeddings_pet.pb', 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        
        global_models['graph'] = tf.compat.v1.get_default_graph()
        
        logger.info("All models initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        return False

def create_db_connection():
    """Create SQL Server connection"""
    try:
        conn_str = (
            f"DRIVER={{SQL Server}};"
            f"SERVER={CONFIG['sql_server']};"
            f"DATABASE={CONFIG['sql_db']};"
            f"UID={CONFIG['sql_username']};"
            f"PWD={CONFIG['sql_password']}"
        )
        return pyodbc.connect(conn_str)
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return None

def get_mongo_client():
    """Create MongoDB client"""
    try:
        return MongoClient(CONFIG['mongodb_uri'])
    except Exception as e:
        logger.error(f"MongoDB connection failed: {str(e)}")
        return None

def verify_image_integrity(image_id, expected_md5=None):
    """Verify image integrity using MD5 checksum and content validation"""
    try:
        client = get_mongo_client()
        if client is None:
            return False
            
        db = client[CONFIG['mongodb_name']]
        fs = GridFS(db, CONFIG['collections']['images'])
        
        if not fs.exists(_id=image_id):
            logger.debug(f"Image {image_id} not found in GridFS")
            return False
            
        # Get the file record to check expected MD5 if provided
        file_record = db.fs.files.find_one({"_id": image_id})
        if expected_md5 and file_record.get("md5") != expected_md5:
            logger.warning(f"MD5 mismatch for image {image_id}")
            return False
            
        # Verify we can read the entire file
        grid_file = fs.get(image_id)
        try:
            img_data = grid_file.read()
            
            # Verify the image content
            try:
                # Method 1: PIL verify
                img = Image.open(io.BytesIO(img_data))
                img.verify()
                
                # Method 2: Try to load the image
                img = Image.open(io.BytesIO(img_data))
                img.load()
                
                # Method 3: Try OpenCV
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("OpenCV failed to decode image")
                    
                return True
            except Exception as verify_error:
                logger.error(f"Image verification failed for {image_id}: {str(verify_error)}")
                return False
        except Exception as read_error:
            logger.error(f"Error reading image {image_id}: {str(read_error)}")
            return False
    except Exception as e:
        logger.error(f"Error verifying image {image_id}: {str(e)}")
        return False
    finally:
        if 'client' in locals():
            client.close()

def fix_corrupted_jpeg(img_data):
    """Attempt to fix corrupted JPEG data"""
    try:
        # Look for JPEG markers
        jpeg_start = img_data.find(b'\xff\xd8')
        jpeg_end = img_data.rfind(b'\xff\xd9')
        
        if jpeg_start != -1 and jpeg_end != -1:
            fixed_data = img_data[jpeg_start:jpeg_end+2]
            
            # Basic validation of the fixed data
            try:
                Image.open(io.BytesIO(fixed_data)).verify()
                return fixed_data
            except:
                return img_data
        return img_data
    except:
        return img_data

def retriving_petimage(image_id):
    """Enhanced image retrieval with multiple fallback methods and better error handling"""
    try:
        client = get_mongo_client()
        if client is None:
            return None
            
        db = client[CONFIG['mongodb_name']]
        fs = GridFS(db, CONFIG['collections']['images'])
        
        # First verify the file exists and is readable
        if not fs.exists(_id=image_id):
            logger.error(f"Image {image_id} not found in GridFS")
            return None
            
        try:
            grid_file = fs.get(image_id)
            img_data = grid_file.read()
            
            # Get the expected MD5 from the files collection
            file_record = db.fs.files.find_one({"_id": image_id})
            expected_md5 = file_record.get("md5") if file_record else None
            
            # Verify MD5 if available
            if expected_md5:
                md5_hash = hashlib.md5(img_data).hexdigest()
                if md5_hash != expected_md5:
                    logger.warning(f"MD5 mismatch for image {image_id}. Expected: {expected_md5}, Got: {md5_hash}")
            
            # Define all our decoding attempts
            decode_attempts = [
                # Attempt 1: OpenCV direct
                lambda: cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR),
                
                # Attempt 2: PIL conversion
                lambda: cv2.cvtColor(np.array(Image.open(io.BytesIO(img_data))), cv2.COLOR_RGB2BGR),
                
                # Attempt 3: Try with different flags
                lambda: cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_ANYCOLOR),
                
                # Attempt 4: Try with fixed JPEG data
                lambda: cv2.imdecode(np.frombuffer(fix_corrupted_jpeg(img_data), np.uint8), cv2.IMREAD_COLOR),
                
                # Attempt 5: Try with different color conversion
                lambda: cv2.cvtColor(cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2BGR)
            ]
            
            for i, attempt in enumerate(decode_attempts):
                try:
                    img = attempt()
                    if img is not None and img.size > 0:
                        logger.debug(f"Successfully decoded image {image_id} with method {i+1}")
                        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    logger.debug(f"Decode method {i+1} failed for image {image_id}: {str(e)}")
                    continue
                    
            logger.error(f"All decode methods failed for image {image_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error reading image {image_id}: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"Error retrieving image {image_id}: {str(e)}")
        return None
    finally:
        if 'client' in locals():
            client.close()

def get_face_embedding(face_img):
    """Get embedding for a single face"""
    try:
        with global_models['graph'].as_default():
            with global_models['session'].as_default():
                resized = cv2.resize(face_img, (160, 160))
                prewhitened = facenet.prewhiten(resized)
                feed_dict = {
                    'input:0': [prewhitened],
                    'phase_train:0': False
                }
                return global_models['session'].run('embeddings:0', feed_dict=feed_dict)
    except Exception as e:
        logger.error(f"Error getting face embedding: {str(e)}")
        return None

def extract_faces(image):
    """Extract all faces from an image"""
    try:
        results = global_models['detector'].detect_face(image)
        if results is None:
            return [], []
            
        total_boxes = results[0]
        points = results[1]
        chips = global_models['detector'].extract_image_chips(image, points, 160, 0.33)
        return chips, total_boxes.tolist()
    except Exception as e:
        logger.error(f"Error extracting faces: {str(e)}")
        return [], []

def decode_pet_regimage(img_data):
    """Decode PET registration image from SQL"""
    try:
        if isinstance(img_data, bytes):
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image bytes")
        else:
            img_data = img_data.replace(" ", "+")
            if ',' in img_data:
                img_data = img_data.split(',')[-1]
            img = cv2.cvtColor(
                np.array(Image.open(io.BytesIO(base64.b64decode(img_data)))),
                cv2.COLOR_BGR2RGB
            )
        return img
    except Exception as e:
        logger.error(f"Error decoding PET image: {str(e)}")
        return None

def get_frames_to_process():
    """Get frames with valid images only"""
    try:
        client = get_mongo_client()
        if client is None:
            return []
            
        db = client[CONFIG['mongodb_name']]
        fs = GridFS(db, CONFIG['collections']['images'])
        
        # Get all available image IDs for faster lookup
        available_images = set(str(f._id) for f in fs.find())
        logger.info(f"Found {len(available_images)} available images in GridFS")
        
        frames = []
        skipped_missing = 0
        
        # Process bib_detection_results
        bib_coll = db[CONFIG['collections']['bib_recognition']]
        bib_query = {
            "image": {"$exists": True, "$ne": None},
            "$or": [
                {"facial_flag": {"$exists": False}},
                {"facial_flag": None},
                {"facial_flag": ""}
            ],
            "rollno": {"$exists": True, "$ne": None}
        }
        
        for doc in bib_coll.find(bib_query):
            if str(doc['image']) in available_images:
                # Verify the image is actually readable
                if verify_image_integrity(doc['image']):
                    doc['_collection'] = 'bib_detection_results'
                    frames.append(doc)
                else:
                    skipped_missing += 1
                    logger.warning(f"Skipping bib doc {doc['_id']} - image {doc['image']} is corrupted")
            else:
                skipped_missing += 1
                logger.debug(f"Skipping bib doc {doc['_id']} - image {doc['image']} not found")
        
        # Process wrong_tag_results
        wrong_coll = db[CONFIG['collections']['wrong_tag_results']]
        wrong_query = {
            "image": {"$exists": True, "$ne": None},
            "$or": [
                {"facial_flag": {"$exists": False}},
                {"facial_flag": None},
                {"facial_flag": ""}
            ]
        }
        
        for doc in wrong_coll.find(wrong_query):
            if str(doc['image']) in available_images:
                # Verify the image is actually readable
                if verify_image_integrity(doc['image']):
                    doc['_collection'] = 'wrong_tag_results'
                    frames.append(doc)
                else:
                    skipped_missing += 1
                    logger.warning(f"Skipping wrong tag doc {doc['_id']} - image {doc['image']} is corrupted")
            else:
                skipped_missing += 1
                logger.debug(f"Skipping wrong tag doc {doc['_id']} - image {doc['image']} not found")
        
        logger.info(f"Found {len(frames)} valid frames (skipped {skipped_missing} with missing/corrupted images)")
        return frames
    except Exception as e:
        logger.error(f"Error getting frames to process: {str(e)}")
        return []
    finally:
        if 'client' in locals():
            client.close()

def process_pet_images():
    """Process all PET images from SQL and save embeddings"""
    try:
        conn = create_db_connection()
        if conn is None:
            return False
            
        cursor = conn.cursor()
        cursor.execute("SELECT roll_no, imagedata_face FROM dbo.imagedata WHERE imagedata_face IS NOT NULL")
        
        pet_data = {}
        logger.info("Processing PET images...")
        processed_count = 0
        failed_count = 0
        
        for roll_no, img_data in cursor:
            try:
                img = decode_pet_regimage(img_data)
                if img is None:
                    failed_count += 1
                    continue
                    
                faces, _ = extract_faces(img)
                if faces:
                    embedding = get_face_embedding(faces[0])
                    if embedding is not None:
                        pet_data[str(roll_no)] = embedding[0]
                        processed_count += 1
            except Exception as e:
                failed_count += 1
                logger.error(f"Error processing PET {roll_no}: {str(e)}")
        
        with open('pet_embeddings.pkl', 'wb') as f:
            pickle.dump(pet_data, f)
            
        logger.info(f"Saved {processed_count} PET embeddings ({failed_count} failed)")
        return True
    except Exception as e:
        logger.error(f"Error in process_pet_images: {str(e)}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def process_mongo_frames():
    """Process all frames from MongoDB and save embeddings"""
    try:
        frames = get_frames_to_process()
        if not frames:
            logger.info("No valid frames to process")
            return False
            
        frame_data = {}
        logger.info("Processing frames from MongoDB...")
        processed_count = 0
        failed_count = 0
        
        for doc in frames:
            try:
                img = retriving_petimage(doc['image'])
                if img is None:
                    failed_count += 1
                    logger.warning(f"Skipping frame {doc['_id']} - image could not be loaded")
                    continue
                    
                faces, bboxes = extract_faces(img)
                if not faces:
                    logger.info(f"No faces detected in frame {doc['_id']}")
                    continue
                    
                embeddings = []
                for face in faces:
                    emb = get_face_embedding(face)
                    if emb is not None:
                        embeddings.append(emb[0])
                
                if embeddings:
                    frame_data[str(doc['_id'])] = {
                        'embeddings': embeddings,
                        'bboxes': bboxes,
                        'metadata': {
                            'rollno': doc.get('rollno'),
                            'bib': doc.get('final_verified_bib') or doc.get('bib_number', ''),
                            'timestamp': doc.get('timestamp'),
                            'lap': doc.get('Lap'),
                            'source_collection': doc.get('_collection', 'unknown'),
                            'ai_unique_id': doc.get('ai_unique_id', '')
                        }
                    }
                    processed_count += 1
            except Exception as e:
                failed_count += 1
                logger.error(f"Error processing frame {doc.get('_id')}: {str(e)}")
        
        with open('frame_embeddings.pkl', 'wb') as f:
            pickle.dump(frame_data, f)
            
        logger.info(f"Saved {processed_count} frame embeddings ({failed_count} failed)")
        return True
    except Exception as e:
        logger.error(f"Error in process_mongo_frames: {str(e)}")
        return False

def update_facial_flags(frame_ids):
    """Update facial_flag in both collections for processed frames"""
    try:
        client = get_mongo_client()
        if client is None:
            return False
            
        db = client[CONFIG['mongodb_name']]
        
        # Split into batches
        for i in range(0, len(frame_ids), CONFIG['batch_size']):
            batch = frame_ids[i:i + CONFIG['batch_size']]
            
            # Update bib_detection_results
            bib_result = db[CONFIG['collections']['bib_recognition']].update_many(
                {'_id': {'$in': batch}},
                {'$set': {'facial_flag': '1'}}
            )
            
            # Update wrong_tag_results
            wrong_result = db[CONFIG['collections']['wrong_tag_results']].update_many(
                {'_id': {'$in': batch}},
                {'$set': {'facial_flag': '1'}}
            )
            
            logger.info(
                f"Updated batch {i//CONFIG['batch_size'] + 1}: "
                f"Bib={bib_result.modified_count}, Wrong={wrong_result.modified_count}"
            )
        
        return True
    except Exception as e:
        logger.error(f"Error updating facial flags: {str(e)}")
        return False
    finally:
        if 'client' in locals():
            client.close()

def match_faces():
    """Match PET faces with frames using pre-computed embeddings"""
    try:
        # Load embeddings
        logger.info("Loading embeddings...")
        try:
            with open('pet_embeddings.pkl', 'rb') as f:
                pet_embeddings = pickle.load(f)
            
            with open('frame_embeddings.pkl', 'rb') as f:
                frame_embeddings = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            return False
        
        client = get_mongo_client()
        if client is None:
            return False
            
        db = client[CONFIG['mongodb_name']]
        
        results = []
        detection_details = []
        processed_frame_ids = []
        
        logger.info("Matching faces...")
        matched_count = 0
        not_matched_count = 0
        
        for frame_id, frame_data in frame_embeddings.items():
            frame_rollno = frame_data['metadata']['rollno']
            if frame_rollno not in pet_embeddings:
                continue
                
            # Vectorized similarity calculation
            similarities = cosine_similarity(
                [pet_embeddings[frame_rollno]],
                frame_data['embeddings']
            )
            best_idx = np.argmax(similarities)
            best_sim = similarities[0][best_idx]
            
            match_status = 'matched' if best_sim > CONFIG['similarity_threshold'] else 'not_matched'
            if match_status == 'matched':
                matched_count += 1
            else:
                not_matched_count += 1
            
            # Prepare result document
            result_doc = {
                'frame_id': frame_id,
                'rollno': frame_rollno,
                'bib_number': frame_data['metadata']['bib'],
                'similarity': float(best_sim),
                'match_status': match_status,
                'source_collection': frame_data['metadata'].get('source_collection', 'unknown'),
                'processed_at': datetime.now(),
                'lap': frame_data['metadata']['lap'],
                'pet_date': frame_data['metadata']['timestamp'].split(' ')[0] if frame_data['metadata']['timestamp'] else None,
                'ai_unique_id': frame_data['metadata'].get('ai_unique_id', ''),
                'original_doc': {
                    'bib_number': frame_data['metadata']['bib'],
                    'rollno': frame_rollno,
                    'timestamp': frame_data['metadata']['timestamp'],
                    'Lap': frame_data['metadata']['lap']
                }
            }
            results.append(result_doc)
            processed_frame_ids.append(frame_id)
            
            # Prepare detection details
            detection_details.append({
                'frame_id': frame_id,
                'rollno': frame_rollno,
                'bib_number': frame_data['metadata']['bib'],
                'pet_date': frame_data['metadata']['timestamp'].split(' ')[0] if frame_data['metadata']['timestamp'] else None,
                'timestamp': frame_data['metadata']['timestamp'],
                'Lap': frame_data['metadata']['lap'],
                'no_of_faces_detected': len(frame_data['embeddings']),
                'best_match_similarity': float(best_sim),
                'best_match_bbox': frame_data['bboxes'][best_idx],
                'source_collection': frame_data['metadata'].get('source_collection', 'unknown'),
                'ai_unique_id': frame_data['metadata'].get('ai_unique_id', ''),
                'processed_at': datetime.now()
            })
        
        # Create new collections if they don't exist
        if CONFIG['collections']['facial_results'] not in db.list_collection_names():
            db.create_collection(CONFIG['collections']['facial_results'])
        
        if CONFIG['collections']['facial_detections'] not in db.list_collection_names():
            db.create_collection(CONFIG['collections']['facial_detections'])
        
        # Bulk insert results
        if results:
            logger.info(f"Inserting {len(results)} facial recognition results ({matched_count} matched, {not_matched_count} not matched)...")
            db[CONFIG['collections']['facial_results']].insert_many(results)
            logger.info(f"Inserted {len(results)} facial recognition results")
        
        if detection_details:
            logger.info(f"Inserting {len(detection_details)} facial detection details...")
            db[CONFIG['collections']['facial_detections']].insert_many(detection_details)
            logger.info(f"Inserted {len(detection_details)} facial detection details")
        
        # Update flags in both source collections
        if processed_frame_ids:
            logger.info("Updating facial flags in source collections...")
            update_facial_flags(processed_frame_ids)
        
        return True
    except Exception as e:
        logger.error(f"Error in match_faces: {str(e)}")
        return False
    finally:
        if 'client' in locals():
            client.close()

def main():
    """Main execution pipeline"""
    if not init_models():
        exit(1)
    
    try:
        logger.info("Starting facial recognition pipeline")
        
        logger.info("Processing PET images from SQL...")
        if not process_pet_images():
            logger.error("PET image processing failed")
            exit(1)
        
        logger.info("Processing frames from MongoDB...")
        if not process_mongo_frames():
            logger.error("Frame processing failed")
            exit(1)
        
        logger.info("Matching faces...")
        if not match_faces():
            logger.error("Face matching failed")
            exit(1)
        
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
    finally:
        # Clean up
        if global_models['session']:
            global_models['session'].close()

if __name__ == "__main__":
    main()