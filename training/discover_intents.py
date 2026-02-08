import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAINING_DIR = os.path.dirname(__file__)

UNRECOGNIZED_FILE = os.path.join(DATA_DIR, "unrecognized.json")
OUTPUT_FILE = os.path.join(TRAINING_DIR, "discovered_intents.txt")

if not os.path.exists(UNRECOGNIZED_FILE):
    print("No unrecognized data yet.")
    exit()

with open(UNRECOGNIZED_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

sentences = [
    item["message"]
    for item in data
    if isinstance(item, dict)
    and "message" in item
    and len(item["message"].split()) > 3
]

if len(sentences) < 5:
    print("Not enough data for intent discovery.")
    exit()

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)

k = min(5, len(sentences))
kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
labels = kmeans.fit_predict(embeddings)

clusters = {}
for sentence, label in zip(sentences, labels):
    clusters.setdefault(label, []).append(sentence)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for i, cluster in clusters.items():
        f.write(f"\nIntent Candidate {i + 1}\n")
        for s in cluster[:5]:
            f.write(f"- {s}\n")

print(f"Discovered intents saved to: {OUTPUT_FILE}")
