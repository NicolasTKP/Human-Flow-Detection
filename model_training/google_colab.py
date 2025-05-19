from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics

import ultralytics
ultralytics.checks()
#Create a file call 'yolov8_custom.yaml' in `content/` if using custom yolo model layers


from ultralytics import YOLO
model = YOLO("yolov8_custom.yaml")
model.info()

model.train(data="/content/drive/My Drive/datasets/data.yaml", epochs=50, imgsz=640, batch=16, device="cuda")

model.val()

import locale
locale.getpreferredencoding = lambda: "UTF-8"

!cp -r /content/runs/ /content/drive/MyDrive/YoloTraining/







model_types = ['n', 's', 'm', 'l', 'x']
image_sizes = [240, 480, 640]
batch_sizes = [4, 8, 12, 16, 32]

# Dataset path (adjust to your actual path)
data_yaml = '/content/drive/My Drive/datasets/data.yaml'

# Store evaluation results
eval_results = {}

# Train and evaluate each combination
for model_type in model_types:
    print(f"\n=== Training YOLOv8{model_type} Models ===")
    model = YOLO(f'yolov8{model_type}.pt')  # load pre-trained model
    
    for img_size in image_sizes:
        for batch_size in batch_sizes:
            print(f"\n--- Training yolov8{model_type} | imgsz={img_size} | batch={batch_size} ---")
            
            # Folder for saving this run's results
            run_name = f'yolov8{model_type}_img{img_size}_bs{batch_size}'

            # Train the model
            model.train(
                data=data_yaml,
                imgsz=img_size,
                batch=batch_size,
                epochs=50,  # or more based on your need
                name=run_name,
                project='grid_search_results',
                exist_ok=True
            )

            # Validate the model
            metrics = model.val()
            
            # Extract mAP50-95 (mean average precision)
            map_5095 = metrics.box.map  # float

            # Save result
            eval_results[(model_type, img_size, batch_size)] = map_5095
            print(f"✔ mAP50-95 for {run_name}: {map_5095:.4f}")

# Find the best configuration
best_config = max(eval_results, key=eval_results.get)
best_score = eval_results[best_config]

print("\n\n=== Best Configuration ===")
print(f"Model: yolov8{best_config[0]}")
print(f"Image Size: {best_config[1]}")
print(f"Batch Size: {best_config[2]}")
print(f"mAP50-95: {best_score:.4f}")

# Optional: Load the best model and evaluate on test set
best_model = YOLO(f'yolov8{best_config[0]}.pt')
best_model.train(
    data=data_yaml,
    imgsz=best_config[1],
    batch=best_config[2],
    epochs=50,
    name='best_model',
    project='grid_search_results',
    exist_ok=True
)

# Evaluate on test set
final_metrics = best_model.val()
print("\nFinal Test Evaluation:")
print(final_metrics)
