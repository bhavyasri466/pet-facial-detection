import cv2
import os
import numpy as np
from pymongo import MongoClient
import gridfs
import face_recognition
import base64
from flask import Flask, render_template_string
from concurrent.futures import ProcessPoolExecutor

# --- Flask App Setup ---
app = Flask(__name__)

# --- MongoDB Setup ---
client = MongoClient('mongodb://localhost:27017/')
db = client['video_frames_db']
fs = gridfs.GridFS(db)
matched_collection = db['matched_frames']

# --- Load All Images from Folder ---
def load_reference_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder_path, filename)
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
    print(f"Loaded {len(images)} reference images from folder '{folder_path}'")
    return images

# --- Match a single face to reference images ---
def match_face_to_refs(face_img, reference_images, tolerance=0.6):
    try:
        face_enc = face_recognition.face_encodings(face_img)
        if not face_enc:
            return None

        for idx, ref_img in enumerate(reference_images):
            ref_encs = face_recognition.face_encodings(ref_img)
            if not ref_encs:
                continue
            if face_recognition.compare_faces([ref_encs[0]], face_enc[0], tolerance=tolerance)[0]:
                return idx
    except Exception as e:
        print("Error matching face:", e)
    return None

# --- Check if person is detected in the frame ---
def detect_person(frame):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes, weights = hog.detectMultiScale(gray, winStride=(8,8))
    return len(boxes) > 0

# --- Process Video with Multiprocessing ---
def process_video(video_path, reference_images):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    frame_interval = int(fps // 5)
    frame_id = 0
    actual_id = 0

    executor = ProcessPoolExecutor()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if actual_id % frame_interval == 0:
            if detect_person(frame):
                face_locations = face_recognition.face_locations(frame)
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    face_crop = frame[top:bottom, left:right]
                    _, face_jpg = cv2.imencode('.jpg', face_crop)
                    face_base64 = base64.b64encode(face_jpg).decode('utf-8')

                    # Async match
                    future = executor.submit(match_face_to_refs, face_crop, reference_images)
                    match_index = future.result()

                    if match_index is not None:
                        frame_name = f"frame_{frame_id:05d}_face_{i}.jpg"
                        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
                        file_id = fs.put(frame_bytes, filename=frame_name)
                        matched_collection.insert_one({
                            "frame_name": frame_name,
                            "frame_id": frame_id,
                            "face_index": i,
                            "actual_frame_index": actual_id,
                            "matched_with_index": match_index,
                            "face_base64": face_base64
                        })
                        print(f"[MATCHED] {frame_name} with reference image {match_index}")
                    else:
                        print(f"[NO MATCH] face_{i} in frame_{frame_id:05d}")
                frame_id += 1
            else:
                print(f"[SKIPPED] No person detected in frame {actual_id}")
        actual_id += 1

    cap.release()
    print("Video processing complete.")

# --- Web App to Visualize Matches ---
@app.route('/')
def show_matches():
    matches = matched_collection.find().limit(100)
    html = """
    <h2>Matched Faces</h2>
    {% for match in matches %}
        <div style="margin:10px;display:inline-block;">
            <p><b>{{ match.frame_name }}</b> matched with image {{ match.matched_with_index }}</p>
            <img src="data:image/jpeg;base64,{{ match.face_base64 }}" width="150">
        </div>
    {% endfor %}
    """
    return render_template_string(html, matches=matches)

# --- Run the pipeline and/or web ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        app.run(debug=True)
    else:
        video_path = "D02_20250503061702.mp4"
        reference_folder = "D:/Facial_pickle_new28052025/FacialusingPicklefile_updated/pet_facial_recognition_mongodb_wrongtag/batch_1_images_20250624_155546/"
        reference_images = load_reference_images_from_folder(reference_folder)
        process_video(video_path, reference_images)
