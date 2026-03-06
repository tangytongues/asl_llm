import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

DATASET_PATH = "custom_asl_dataset"
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10

print("Loading dataset...")

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

num_classes = train_generator.num_classes

print("Classes:", train_generator.class_indices)

print("\nBuilding CNN model...")

model = Sequential([
    Conv2D(32,(3,3),activation="relu",input_shape=(IMG_SIZE,IMG_SIZE,3)),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128,(3,3),activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(256,activation="relu"),
    Dropout(0.5),

    Dense(num_classes,activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("\nTraining model...")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/custom_asl_model.h5"

model.save(MODEL_PATH)

print("\nTraining complete")
print("Model saved to:", MODEL_PATH)