from google.colab import drive
drive.mount('/content/drive')

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  

model.train(data="/content/drive/My Drive/datasets/data.yaml", epochs=50, imgsz=640, batch=16, device="cuda")

model.val()

import locale
locale.getpreferredencoding = lambda: "UTF-8"

!cp -r /content/runs/ /content/drive/MyDrive/YoloTraining/