import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing import image_dataset_from_directory
import cv2
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def get_new_model(input_shape):
    '''
    This function returns a compiled CNN with specifications given above.
    '''

    # Defining the architecture of the CNN
    input_layer = Input(shape=input_shape, name='input')
    h = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_1')(input_layer)
    h = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_2')(h)

    h = MaxPool2D(pool_size=(2,2), name='pool_1')(h)

    h = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_3')(h)
    h = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_4')(h)

    h = MaxPool2D(pool_size=(2,2), name='pool_2')(h)

    h = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_5')(h)
    h = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_6')(h)

    h = Flatten(name='flatten_1')(h)  # Flatten first
    h = Dense(64, activation='relu', name='dense_1')(h)  # Now apply Dense layer
    h = Dropout(0.5, name='dropout_1')(h)

    output_layer = Dense(10, activation='softmax', name='dense_2')(h)

    # Generate the model
    model = Model(inputs=input_layer, outputs=output_layer, name='model_CNN')

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


dataset_path = "C:/Users/acer/Desktop/Python/Python/Human-Flow-Detection/datasets/images/train"
val_path = "C:/Users/acer/Desktop/Python/Python/Human-Flow-Detection/datasets/images/val"
# Define image size
IMG_SIZE = (128, 128)  # Change based on your model input shape

# Get all image file paths
image_paths = [os.path.join(dataset_path, fname) for fname in os.listdir(dataset_path) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
val_paths = [os.path.join(val_path, fname) for fname in os.listdir(val_path) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]


def load_and_preprocess_image(image_path):
    image_path = image_path.numpy().decode('utf-8')  # Convert Tensor to string
    img = load_img(image_path, target_size=IMG_SIZE)  # Load and resize image
    img = img_to_array(img) / 255.0  # Convert to array and normalize
    return img

# Convert images into TensorFlow dataset
image_data = tf.data.Dataset.from_tensor_slices(image_paths)
image_data = image_data.map(lambda x: tf.py_function(func=load_and_preprocess_image, inp=[x], Tout=tf.float32))
val_data = tf.data.Dataset.from_tensor_slices(val_paths)
val_data = val_data.map(lambda x: tf.py_function(func=load_and_preprocess_image, inp=[x], Tout=tf.float32))


# Batch and shuffle the dataset
BATCH_SIZE = 32
train_ds = image_data.batch(BATCH_SIZE).shuffle(buffer_size=len(image_paths))
val_ds = val_data.batch(BATCH_SIZE).shuffle(buffer_size=len(val_paths))



# Check dataset output shape
for img_batch in train_ds.take(1):
    print(img_batch.shape)

input_shape = (128, 128, 3)
model = get_new_model(input_shape)

# Remove the output layer and extract features from the last conv layer
feature_extractor = Model(
    inputs=model.input, 
    outputs=model.get_layer("conv2d_6").output  # Extracts features from last conv layer
)

# Compile the feature extractor model
feature_extractor.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
feature_extractor.fit(train_ds, validation_data=val_ds, epochs=10)


img_path = "datasets/images/train/1.png"
img = cv2.imread(img_path)
img = img / 255.0  # Normalize
img = np.expand_dims(img, axis=0)

res = feature_extractor.predict(img)  # Extract features from the last conv layer
print(res)