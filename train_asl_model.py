import os
import cv2
import numpy as np
import kagglehub
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ============================
# CONFIG (Safe for 8GB RAM)
# ============================

IMG_SIZE = 64
MAX_IMAGES = 20000
BATCH_SIZE = 32
EPOCHS = 5

# ============================
# DOWNLOAD DATASET
# ============================

print("Downloading ASL dataset from Kaggle...")

dataset_path = kagglehub.dataset_download("grassknoted/asl-alphabet")

dataset_dir = os.path.join(dataset_path, "asl_alphabet_train", "asl_alphabet_train")

print("Dataset location:", dataset_dir)

labels = sorted(os.listdir(dataset_dir))
label_map = {label: i for i, label in enumerate(labels)}

print("Classes detected:", labels)
print("Total classes:", len(labels))

# ============================
# LOAD DATASET
# ============================

X = []
y = []

print("\nLoading images...")

image_count = 0

for label in labels:

    folder = os.path.join(dataset_dir, label)

    for img_name in tqdm(os.listdir(folder), desc=f"Loading {label}"):

        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        X.append(img)
        y.append(label_map[label])

        image_count += 1

        if image_count >= MAX_IMAGES:
            break

    if image_count >= MAX_IMAGES:
        break


X = np.array(X, dtype=np.float32)

# IMPORTANT FIX: force full class size
y = to_categorical(np.array(y), num_classes=len(labels))

print("\nDataset loaded successfully")
print("Total samples:", X.shape)

# ============================
# TRAIN / VALIDATION SPLIT
# ============================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))

# ============================
# BUILD CNN MODEL
# ============================

print("\nBuilding CNN model...")

model = Sequential([

    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(len(labels), activation='softmax')

])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================
# TRAIN MODEL
# ============================

print("\nTraining model...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ============================
# SAVE MODEL
# ============================

os.makedirs("models", exist_ok=True)

model_path = "models/asl_cnn.h5"

model.save(model_path)

print("\nTraining complete.")
print("Model saved to:", model_path)