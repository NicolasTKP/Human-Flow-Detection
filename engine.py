from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchreid.reid.utils import FeatureExtractor
from scipy.spatial.distance import cosine
from bs4 import BeautifulSoup
import threading
import time

# Load model and parameters
with open('./parameter.xml', 'r') as f:
    xml = f.read()
Bs_data = BeautifulSoup(xml, "xml")

inference_threshold = float(Bs_data.find('inference_threshold').text)
feature_extraction_threshold = float(Bs_data.find('feature_extraction_threshold').text)
print(f"Thresholds: {inference_threshold}, {feature_extraction_threshold}")

model = YOLO(".\\runs\\detect\\train\\weights\\best.pt")

extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path=None,  # If using a pretrained model
    device='cpu'  # Change to 'cpu' if GPU is unavailable
)

person_embeddings = {}  # Format: {track_id: (embedding, timestamp, cam_id)}
embedding_lock = threading.Lock() #To ensure only one thread accesses the embeddings at a time
person_last_seen = {}  # {track_id: (last_cam_id, timestamp)}
total_transitions = 0

def process_camera(cam_id, camera_index):
    global total_transitions
    tracker = DeepSort(max_age=1, embedder="torchreid", embedder_gpu=True)
    cap = cv2.VideoCapture(camera_index) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #Set the reslution of the camera
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=inference_threshold)  # Threshold for confidence score
        if len(results) == 0 or len(results[0].boxes) == 0:
            cv2.putText(frame, f"Human Detected: {len(person_embeddings)}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Cross-Zone: {total_transitions}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow(f"Human Tracking Cam {cam_id}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        detections = []
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.tolist())
                conf = float(result.boxes.conf[0])
                cls = int(result.boxes.cls[0])

                if cls == 0:
                    detections.append(([x1, y1, x2, y2], conf, "human"))

        tracked_objects = tracker.update_tracks(detections, frame=frame)
        frame_with_yolo = results[0].plot()

        for track in tracked_objects:
            if track.is_confirmed():
                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_tlbr())

                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size != 0:
                    embedding = extractor(person_crop)
                    embedding = embedding[0]

                    best_match_id = None
                    best_similarity = float("inf")

                    with embedding_lock:
                        for existing_id, stored_embedding in person_embeddings.items():
                            similarity = cosine(embedding, stored_embedding)
                            if similarity < feature_extraction_threshold: # Threshold for similarity (lower = more similar)
                                best_match_id = existing_id
                                best_similarity = similarity

                        if best_match_id is not None:
                            track_id = best_match_id
                        else:
                            person_embeddings[track_id] = embedding
                        
                        # Add total_transitions by 1 if the person is seen in a different camera
                        current_time = time.time()
                        if track_id in person_last_seen:
                            last_cam_id, last_seen_time = person_last_seen[track_id]
                            if last_cam_id != cam_id:
                                total_transitions += 1

                        # Update last seen info
                        person_last_seen[track_id] = (cam_id, current_time)

                cv2.putText(frame_with_yolo, f"ID: {track_id}", (x1+150, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(frame_with_yolo, f"Human Detected: {len(person_embeddings)}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame_with_yolo, f"Cross-Zone: {total_transitions}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow(f"Human Tracking Cam {cam_id}", frame_with_yolo)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    thread0 = threading.Thread(target=process_camera, args=(0, 0))
    thread1 = threading.Thread(target=process_camera, args=(1, 1))

    thread0.start()
    thread1.start()

    thread0.join()
    thread1.join()
