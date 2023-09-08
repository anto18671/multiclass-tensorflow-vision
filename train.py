import os
import numpy as np
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
data_path = "C:/Users/Anthony/Desktop/multiclass-tensorflow-vision/dataset/merged"

# Get a list of all images and their respective classes
all_images = []
all_labels = []
for class_name in os.listdir(data_path):
    class_path = os.path.join(data_path, class_name)
    if os.path.isdir(class_path):
        for filename in os.listdir(class_path):
            all_images.append(os.path.join(class_name, filename))
            all_labels.append(class_name)

# Convert lists to arrays for shuffling
all_images = np.array(all_images)
all_labels = np.array(all_labels)

# Shuffle both arrays
indices = np.arange(all_images.shape[0])
np.random.shuffle(indices)
all_images = all_images[indices]
all_labels = all_labels[indices]

# Splitting data using ImageDataGenerator
datagen = ImageDataGenerator(
    validation_split=0.10,
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Create data generators
train_generator = datagen.flow_from_directory(
    directory=data_path,
    subset="training",
    target_size=(224, 224),
    batch_size=28,
    class_mode="categorical",
)

validation_generator = datagen.flow_from_directory(
    directory=data_path,
    subset="validation",
    target_size=(224, 224),
    batch_size=28,
    class_mode="categorical",
)

# Model using Sequential API
model = Sequential([
    EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    ),
    GlobalAveragePooling2D(),
    Dense(1024, activation="relu"),
    Dense(1024, activation="relu"),
    Dense(train_generator.num_classes, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=CategoricalCrossentropy(),
    metrics=[CategoricalAccuracy()],
)

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    verbose=1,
)