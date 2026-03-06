from joblib import load

model = load("model/gesture_model.pkl")

print("Classes inside model:")
print(model.classes_)