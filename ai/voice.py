import pyttsx3


def speak(text: str) -> None:
    """
    Synchronous text-to-speech helper.
    Re-initializes the engine per call for reliability on some systems.
    """
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    engine.setProperty("volume", 1.0)
    engine.say(text)
    engine.runAndWait()
    engine.stop()