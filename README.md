# BOT1 — Hand Gesture AI Assistant (ASL + Commands)

BOT1 is a single-process, real-time desktop assistant built in Python. It uses your webcam to recognize **hand gestures** for fast commands and (optionally) **ASL alphabet letters** for typing, then speaks responses using offline TTS.

## Features

- **Realtime webcam loop** (OpenCV)
- **Hand tracking** (MediaPipe)
- **FAST command mode** (scikit-learn gesture classifier)
- **ALPHABET typing mode** (TensorFlow/Keras CNN, optional)
- **Text-to-speech** (pyttsx3, blocking/synchronous)
- **Fullscreen OpenCV HUD** (clean UI panels + center indicator)

## Quick start

### 1) Create + activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

If you use the **ALPHABET mode**, install TensorFlow:

```powershell
pip install tensorflow
```

### 3) Run

```powershell
python main.py
```

Press **ESC** to exit.

## Operating modes

BOT1 supports two modes:

- **FAST**: quick command gestures (HELLO, HELP, LIGHT_ON, etc.)
- **ALPHABET**: ASL alphabet letter recognition → word/sentence building → optional LLM reply

Mode switching (when enabled in your build) is done via a **MODE_SWITCH** gesture.

## FAST mode gestures (typical)

Depending on your trained model/classes, your FAST classifier may recognize:

- `HELLO`
- `THANK_YOU`
- `HELP`
- `LIGHT_ON`
- `LIGHT_OFF`
- `MODE_SWITCH` (if included in the model)

### Light confirmation

After `LIGHT_ON` / `LIGHT_OFF` is detected, BOT1 enters a `CONFIRM` state with a pending action.

In the current implementation, confirmation can be done via:

- **Confirm**: `THANK_YOU` **or** `THUMBS_UP`
- **Cancel**: `HELP`
- *(Optional fallback)* swipe right to confirm / swipe left to cancel (if enabled)

## ALPHABET mode (optional)

If present in your project:

- Model file: `models/asl_cnn.h5`
- Inference helper: `asl/alphabet_model.py`
- Word building: `asl/word_builder.py`

Typical classes include: `A-Z`, `del`, `space`, `nothing`.

## Project structure (high level)

- `main.py`: main webcam loop + state machine
- `vision/`: MediaPipe tracker + feature extractor
- `model/`: gesture classifier inference
- `logic/`: swipe/confirmation logic
- `ai/`: TTS + (optional) local LLM integration
- `ui/`: OpenCV HUD rendering
- `asl/`: ASL alphabet CNN inference + word builder (if used)

## Troubleshooting

### MediaPipe protobuf crash (AttributeError FieldDescriptor / GetPrototype)

If you see errors like:

- `MessageFactory object has no attribute GetPrototype`
- `FieldDescriptor object has no attribute label`

your `protobuf` is too new for your installed `mediapipe`. Fix by pinning protobuf:

```powershell
pip install "protobuf==3.20.3" --upgrade
```

### No audio / pyttsx3 not speaking

- Ensure you’re running on Windows with an available system voice engine.
- Try running a minimal test:

```python
from ai.voice import speak
speak("BOT1 voice test.")
```

## Notes

- This repo may contain large assets (datasets/models). Consider using **Git LFS** for big binary files if needed.

