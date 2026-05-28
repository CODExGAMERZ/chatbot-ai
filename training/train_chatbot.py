import json
import random
import numpy as np
import nltk
import pickle
import os
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

INTENTS_FILE = os.path.join(DATA_DIR, "intents.json")
WORDS_FILE = os.path.join(MODEL_DIR, "words.pkl")
CLASSES_FILE = os.path.join(MODEL_DIR, "classes.pkl")
MODEL_FILE = os.path.join(MODEL_DIR, "chatbot_model.h5")

with open(INTENTS_FILE, "r") as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open(WORDS_FILE, "wb"))
pickle.dump(classes, open(CLASSES_FILE, "wb"))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]

    for word in words:
        bag.append(1 if word in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

X = np.array(list(training[:, 0]))
y = np.array(list(training[:, 1]))

model = Sequential()
model.add(Dense(128, input_shape=(len(X[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dense(len(y[0]), activation="softmax"))

model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.01),
    metrics=["accuracy"]
)

model.fit(X, y, epochs=200, batch_size=5, verbose=1)

model.save(MODEL_FILE)
print("Chatbot training complete")
