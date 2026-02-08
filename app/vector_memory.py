import faiss
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

INDEX_FILE = os.path.join(DATA_DIR, "vector.index")
META_FILE = os.path.join(DATA_DIR, "vector_meta.json")

model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384

if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    index = faiss.IndexFlatL2(dimension)

if os.path.exists(META_FILE):
    with open(META_FILE, "r") as f:
        memory_meta = json.load(f)
else:
    memory_meta = []

def save_all():
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w") as f:
        json.dump(memory_meta, f, indent=2)

def importance_score(text):
    return min(1.0, 0.4 + len(text.split()) / 15)

def add_to_memory(text):
    embedding = model.encode([text])
    index.add(np.array(embedding).astype("float32"))
    memory_meta.append({
        "text": text,
        "importance": importance_score(text),
        "created": datetime.now().isoformat(),
        "last_used": datetime.now().isoformat(),
        "uses": 1
    })
    save_all()

def search_memory(query, threshold=0.6):
    if index.ntotal == 0:
        return None

    q = model.encode([query])
    distances, indices = index.search(np.array(q).astype("float32"), 1)

    if distances[0][0] < threshold:
        i = indices[0][0]
        memory_meta[i]["uses"] += 1
        memory_meta[i]["last_used"] = datetime.now().isoformat()
        save_all()
        return memory_meta[i]["text"]

    return None

def add_or_merge_memory(text, threshold=0.6):
    hit = search_memory(text, threshold)
    if hit:
        return hit
    add_to_memory(text)
    return text
