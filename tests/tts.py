
import pyttsx3

engine = pyttsx3.init(driverName="sapi5")
voices = engine.getProperty("voices")

engine.setProperty("voice", voices[0].id)
engine.setProperty("rate", 170)

engine.say("If you hear this, text to speech is working.")
engine.runAndWait()

print("TTS finished")
