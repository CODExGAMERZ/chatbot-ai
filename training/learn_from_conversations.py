import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
UNREC_FILE = os.path.join(BASE_DIR, "data", "unrecognized.json")

with open(UNREC_FILE, "r") as f:
    data = json.load(f)

sentences = [item["message"] for item in data]

if len(sentences) < 3:
    print("Not enough data to learn yet")
    exit()

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(sentences)

k = min(5, len(sentences))
model = KMeans(n_clusters=k, random_state=42)
model.fit(X)

clusters = {}
for sentence, label in zip(sentences, model.labels_):
    clusters.setdefault(label, []).append(sentence)

for cluster, texts in clusters.items():
    print("\nCLUSTER", cluster)
    for t in texts:
        print(" -", t)
