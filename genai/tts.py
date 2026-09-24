import pyttsx3


def speak(text):
    """
    Convert text into spoken audio.
    """

    engine = pyttsx3.init()

    engine.say(text)
    engine.runAndWait()


if __name__ == "__main__":

    test_text = "The area ahead is crowded, so proceed carefully."

    print("Speaking:")
    print(test_text)

    speak(test_text)