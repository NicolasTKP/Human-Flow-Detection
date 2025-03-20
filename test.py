from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchreid.reid.utils import FeatureExtractor
from scipy.spatial.distance import cosine


model = YOLO(".\\runs\\detect\\train\\weights\\best.pt")

tracker = DeepSort(max_age=1, embedder="torchreid", embedder_gpu=True)

extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path=None,  # If using a pretrained model
    device='cpu'  # Change to 'cpu' if GPU is unavailable
)

person_embeddings = {}

cap = cv2.VideoCapture(0)  
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)
    if len(results) == 0 or len(results[0].boxes) == 0:
        cv2.putText(frame, f"Human Detected: {len(person_embeddings)}", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Human Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    detections = []
    for result in results:
        for box in result.boxes.xyxy:  
            x1, y1, x2, y2 = map(int, box.tolist()) # Bounding box coordinates
            conf = float(result.boxes.conf[0])  # Confidence score
            cls = int(result.boxes.cls[0])  # Class index

            if cls == 0:  # Class 0
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
                embedding = embedding[0]  # convert to a 1D array

                best_match_id = None
                best_similarity = float("inf") # set to infinity

                for existing_id, stored_embedding in person_embeddings.items():
                    similarity = cosine(embedding, stored_embedding)
                    if similarity < 0.45:  # Threshold for cosine similarity (smaller = more similar)
                        best_match_id = existing_id
                        best_similarity = similarity

                if best_match_id is not None:
                    track_id = best_match_id 
                else:
                    person_embeddings[track_id] = embedding  

            cv2.putText(frame_with_yolo, f"ID: {track_id}", (x1+150, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame_with_yolo, f"Human Detected: {len(person_embeddings)}", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Human Tracking", frame_with_yolo)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
