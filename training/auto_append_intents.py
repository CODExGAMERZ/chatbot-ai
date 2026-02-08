import json
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAINING_DIR = os.path.dirname(__file__)

INTENTS_FILE = os.path.join(DATA_DIR, "intents.json")
DISCOVERED_FILE = os.path.join(TRAINING_DIR, "discovered_intents.txt")

if not os.path.exists(DISCOVERED_FILE):
    print("No discovered_intents.txt found.")
    exit()

with open(INTENTS_FILE, "r", encoding="utf-8") as f:
    intents_data = json.load(f)

existing_tags = {intent["tag"] for intent in intents_data["intents"]}

with open(DISCOVERED_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_intents = []
current_patterns = []
intent_index = 1

for line in lines:
    line = line.strip()

    if line.lower().startswith("intent candidate"):
        if current_patterns:
            tag = f"auto_intent_{intent_index}"
            if tag not in existing_tags:
                new_intents.append({
                    "tag": tag,
                    "patterns": current_patterns,
                    "responses": []
                })
                existing_tags.add(tag)
                intent_index += 1
            current_patterns = []
    elif line.startswith("- "):
        phrase = re.sub(r"[^\w\s]", "", line[2:].lower())
        if phrase:
            current_patterns.append(phrase)

if current_patterns:
    tag = f"auto_intent_{intent_index}"
    if tag not in existing_tags:
        new_intents.append({
            "tag": tag,
            "patterns": current_patterns,
            "responses": []
        })

if not new_intents:
    print("No new intents to add.")
    exit()

intents_data["intents"].extend(new_intents)

with open(INTENTS_FILE, "w", encoding="utf-8") as f:
    json.dump(intents_data, f, indent=2)

print(f"Added {len(new_intents)} new intents to intents.json")
