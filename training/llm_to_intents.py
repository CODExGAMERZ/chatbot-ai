import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

LLM_FILE = os.path.join(DATA_DIR, "llm_memory.json")
INTENTS_FILE = os.path.join(DATA_DIR, "intents.json")

with open(LLM_FILE, "r") as f:
    llm_data = json.load(f)

with open(INTENTS_FILE, "r") as f:
    intents = json.load(f)

existing_patterns = {
    p for intent in intents["intents"] for p in intent["patterns"]
}

added = 0

for item in llm_data:
    q = item["question"].lower()
    a = item["answer"]

    if q in existing_patterns:
        continue

    intents["intents"].append({
        "tag": f"llm_learned_{added}",
        "patterns": [q],
        "responses": [a]
    })

    existing_patterns.add(q)
    added += 1

with open(INTENTS_FILE, "w") as f:
    json.dump(intents, f, indent=2)

print(f"Converted {added} LLM answers into intents.")
