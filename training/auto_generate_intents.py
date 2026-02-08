import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
UNREC_FILE = os.path.join(BASE_DIR, "data", "unrecognized.json")

with open(UNREC_FILE, "r") as f:
    data = json.load(f)

sentences = [item["message"].lower() for item in data]

if len(sentences) < 3:
    print("Not enough data to generate intents")
    exit()

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(sentences)

k = min(5, len(sentences))
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X)

clusters = {}
for sentence, label in zip(sentences, labels):
    clusters.setdefault(label, []).append(sentence)

print("\nAUTO-GENERATED INTENT DRAFTS\n")

for i, patterns in clusters.items():
    tag = f"auto_intent_{i}"
    intent = {
        "tag": tag,
        "patterns": patterns,
        "responses": ["Add a proper response here"]
    }
    print(json.dumps(intent, indent=2))
