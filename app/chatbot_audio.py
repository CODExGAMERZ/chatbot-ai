import speech_recognition as sr
import pyttsx3
import time
from app import chatbot_core as chatbot
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

engine = pyttsx3.init(driverName="sapi5")
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[0].id)
engine.setProperty("rate", 170)
engine.setProperty("volume", 1.0)

recognizer = sr.Recognizer()

def speak(text):
    engine.stop()
    engine.say(text)
    engine.runAndWait()
    time.sleep(0.2)

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

print("Voice chatbot started. Say quit or exit to stop.")

while True:
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=6)

    try:
        message = recognizer.recognize_google(audio)
        print("You:", message)

        if message.lower() in ["quit", "exit", "stop"]:
            speak("Goodbye")
            break

        lang = detect_language(message)

        if lang != "en":
            response = "I currently understand only English. Please speak in English."
            speak(response)
            chatbot.save_unrecognized(message)
            chatbot.save_history(message, response, 0.0)
            continue

        tag, confidence = chatbot.predict_class(message)

        if tag is None:
            response = "I'm not sure I understand. Please repeat."
            chatbot.save_unrecognized(message)
        else:
            response = chatbot.get_response(tag)

        print("Bot:", response)
        speak(response)
        chatbot.save_history(message, response, confidence)

    except sr.UnknownValueError:
        speak("Sorry, I didn't catch that.")
