import os
import csv
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import dump

X = []
y = []

data_path = "data/dataset"

for file in os.listdir(data_path):
    label = file.replace(".csv", "")
    with open(os.path.join(data_path, file)) as f:
        reader = csv.reader(f)
        for row in reader:
            X.append([float(val) for val in row])
            y.append(label)

X = np.array(X)

model = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=800
    )
)

model.fit(X, y)

os.makedirs("model", exist_ok=True)
dump(model, "model/gesture_model.pkl")

print("Model trained and saved successfully.")