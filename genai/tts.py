import pyttsx3


def speak(text):
    engine = pyttsx3.init()

    engine.setProperty("rate", 175)
    engine.setProperty("volume", 1.0)

    engine.say(text)
    engine.runAndWait()

    engine.stop()


if __name__ == "__main__":
    test_text = "Obstacle on the left. Move right carefully."

    print("Speaking:")
    print(test_text)

    speak(test_text)