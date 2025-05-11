import cv2

# Open video capture (use the appropriate webcam index, usually 0 or 1)
cap = cv2.VideoCapture(1)  # Change this if your phone is recognized differently

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Show the frame
    cv2.imshow("Phone Camera Feed", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
