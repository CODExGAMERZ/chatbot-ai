import json
from collections import Counter

with open("../data/unrecognized.json", "r") as f:
    data = json.load(f)

phrases = [item["message"].lower() for item in data]

keywords = Counter()
for p in phrases:
    for w in p.split():
        if len(w) > 4:
            keywords[w] += 1

suggested = [k for k, v in keywords.items() if v >= 3]

with open("suggested_intents.txt", "w") as f:
    for word in suggested:
        f.write(word + "\n")
