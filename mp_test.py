import mediapipe as mp

print("Version:", mp.__version__)
print("Has solutions:", hasattr(mp, "solutions"))
print("Hands available:", hasattr(mp.solutions, "hands"))