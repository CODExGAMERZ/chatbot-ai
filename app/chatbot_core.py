import random
import json
import pickle
import numpy as np
import nltk
import datetime
import os

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator

from app.vector_memory import add_or_merge_memory, search_memory
from app.llm_fallback import ask_llm    

DetectorFactory.seed = 0

personality = "friendly"

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()
translator = GoogleTranslator(source="auto", target="en")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

INTENTS_FILE = os.path.join(DATA_DIR, "intents.json")
HISTORY_FILE = os.path.join(DATA_DIR, "chat_history.json")
UNRECOGNIZED_FILE = os.path.join(DATA_DIR, "unrecognized.json")
KNOWLEDGE_FILE = os.path.join(DATA_DIR, "knowledge.json")
LLM_MEMORY_FILE = os.path.join(DATA_DIR, "llm_memory.json")

MODEL_FILE = os.path.join(MODEL_DIR, "chatbot_model.h5")
WORDS_FILE = os.path.join(MODEL_DIR, "words.pkl")
CLASSES_FILE = os.path.join(MODEL_DIR, "classes.pkl")

with open(INTENTS_FILE, "r", encoding="utf-8") as f:
    intents = json.load(f)

with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
    knowledge = json.load(f)

words = pickle.load(open(WORDS_FILE, "rb"))
classes = pickle.load(open(CLASSES_FILE, "rb"))
model = load_model(MODEL_FILE)

conversation_state = {
    "last_user_message": None
}

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def translate_to_english(text):
    try:
        return translator.translate(text)
    except:
        return text

def clean_up_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in tokens]

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, threshold=0.25):
    bow = bag_of_words(sentence)
    probs = model.predict(np.array([bow]), verbose=0)[0]
    sorted_probs = np.sort(probs)
    confidence = float(sorted_probs[-1] - sorted_probs[-2])
    index = np.argmax(probs)
    if confidence < threshold:
        return None, confidence
    return classes[index], confidence

def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

def knowledge_lookup(text):
    for key in knowledge:
        if key in text.lower():
            return knowledge[key]
    return None

def apply_personality(text):
    if personality == "friendly":
        return text + " 😊"
    if personality == "playful":
        return "😄 " + text
    return text

def save_history(user_msg, bot_msg, confidence):
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    except:
        history = []

    history.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "user": user_msg,
        "bot": bot_msg,
        "confidence": round(float(confidence), 2)
    })

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def save_unrecognized(message):
    try:
        with open(UNRECOGNIZED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        data = []

    data.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "message": message
    })

    with open(UNRECOGNIZED_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def save_llm_example(question, answer):
    try:
        with open(LLM_MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        data = []

    data.append({
        "question": question,
        "answer": answer,
        "timestamp": datetime.datetime.now().isoformat()
    })

    with open(LLM_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def run_text_chatbot():
    print("Chatbot is running. Type quit or exit to stop.")

    while True:
        message = input("You: ").strip()

        if message.lower() in ["quit", "exit", "stop"]:
            print("Bot: Goodbye!")
            break

        original_message = message
        conversation_state["last_user_message"] = message

        lang = detect_language(message)
        processed = message if lang == "en" else translate_to_english(message)

        semantic_hit = search_memory(processed)
        if semantic_hit:
            response_en = semantic_hit
            confidence = 1.0
        else:
            fact = knowledge_lookup(processed)
            if fact:
                response_en = fact
                confidence = 1.0
            else:
                tag, confidence = predict_class(processed)
                if tag:
                    response_en = get_response(tag)
                else:
                    response_en = ask_llm(processed)
                    save_llm_example(processed, response_en)
                    confidence = 0.4

        if confidence > 0.75 and len(processed.split()) > 3:
            add_or_merge_memory(processed)

        response_en = apply_personality(response_en)

        if lang == "hi":
            response = GoogleTranslator(source="auto", target="hi").translate(response_en)
        else:
            response = response_en

        print(f"Bot: {response} (confidence: {confidence:.2f})")
        save_history(original_message, response, confidence)

if __name__ == "__main__":
    run_text_chatbot()
