import json
import os
import spacy

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
GRAPH_FILE = os.path.join(DATA_DIR, "knowledge_graph.json")

nlp = spacy.load("en_core_web_sm")

if os.path.exists(GRAPH_FILE):
    with open(GRAPH_FILE, "r") as f:
        graph = json.load(f)
else:
    graph = []

def extract_facts(text):
    doc = nlp(text)
    facts = []

    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            subject = [w.text for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
            obj = [w.text for w in token.rights if w.dep_ in ("dobj", "attr", "prep")]

            if subject and obj:
                facts.append({
                    "subject": subject[0],
                    "relation": token.text,
                    "object": obj[0]
                })

    return facts

def add_to_graph(text):
    facts = extract_facts(text)
    if not facts:
        return

    graph.extend(facts)
    with open(GRAPH_FILE, "w") as f:
        json.dump(graph, f, indent=2)

def query_graph(term):
    results = []
    for fact in graph:
        if term.lower() in fact["subject"].lower():
            results.append(fact)
    return results
