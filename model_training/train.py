from ultralytics import YOLO

model = YOLO("yolov8n.pt")  

model.train(data="data.yaml", epochs=50, imgsz=640, batch=16, device="cuda")

model.val()

# results = model("test_image.jpg", save=True, conf=0.5)
# results.show()