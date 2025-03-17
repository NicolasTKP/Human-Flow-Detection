import kagglehub
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2

path = r"C:\Users\USER\.cache\kagglehub\datasets\constantinwerner\human-detection-dataset\versions\5\human detection dataset\0"
files = os.listdir(path)


for img_file in files[:5]:  # Show first 5 images
    img_path = os.path.join(path, img_file)
    image = cv2.imread(img_path)
    cv2.imshow("Image", image)  # Display image
    cv2.waitKey(1000)  # Wait for 1 second (1000 ms)
    cv2.destroyAllWindows()  # Close the window


for img_file in files[:5]:  # Show first 5 images
    img_path = os.path.join(path, img_file)
    image = Image.open(img_path)

    plt.imshow(image)
    plt.axis("off")  # Hide axes
    plt.show()