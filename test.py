from ultralytics import YOLO
import cv2

model = YOLO(".\\runs\\detect\\train\\weights\\best.pt")
# results = model("186.png", save=True, conf=0.5)


cap = cv2.VideoCapture(0) 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5) #predict

    cv2.imshow("Human Detection", results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()