# 🤖 Hybrid AI Chatbot with Semantic Memory & LLM Learning

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![AI](https://img.shields.io/badge/AI-Hybrid%20Chatbot-purple)
![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-orange)
![Status](https://img.shields.io/badge/status-active-success)

A **production‑style hybrid AI chatbot** that combines classical NLP, machine‑learning intent classification, **semantic vector memory**, and an **LLM fallback used strictly as a teacher**.

The system is designed to **improve over time** by learning from real conversations, discovering new intents, and retraining safely — without blindly trusting LLM outputs.

---

## 🎯 Project Goals

* Build a **realistic AI assistant architecture** (not a toy chatbot)
* Minimize LLM usage while still benefiting from it
* Enable **offline semantic memory** and fast local inference
* Create a **safe learning loop** from conversations
* Follow professional ML + Git practices

---

## ✨ Key Features

* Intent‑based chatbot (fast, local, inexpensive)
* Semantic vector memory using **FAISS** (offline & persistent)
* Knowledge base lookup for deterministic answers
* Hinglish → English auto‑translation
* LLM fallback **only when the bot fails**
* LLM answers saved and reused for training
* Automatic intent discovery from conversations
* Safe retraining pipeline (no blind auto‑learning)
* Memory importance scoring & controlled forgetting

This is a **hybrid AI architecture**, similar to how real assistants are built in production systems.

---

## 🧠 How the Chatbot Thinks (Decision Order)

1. **Semantic Vector Memory (FAISS)**
2. **Knowledge Base Lookup**
3. **Intent Classification Model**
4. **LLM Fallback (Teacher Mode)**

The LLM is **never always‑on**. It is only used when the system cannot confidently respond.

---

## 📁 Project Structure

```
chatbot-ai/
│
├── app/                # Runtime chatbot logic
│   ├── chatbot_core.py
│   ├── vector_memory.py
│   ├── llm_fallback.py
│   ├── knowledge_graph.py
│   └── __init__.py
│
├── training/           # Offline learning & improvement
│   ├── train_chatbot.py
│   ├── discover_intents.py
│   ├── auto_append_intents.py
│   ├── auto_generate_responses.py
│   └── llm_to_intents.py
│
├── data/               # Knowledge & memory (partially git‑ignored)
│   ├── intents.json
│   ├── knowledge.json
│   └── README.md
│
├── model/              # Trained models (generated locally)
│   └── README.md
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠 Installation

```bash
python -m pip install -r requirements.txt
```

Some components require an additional spaCy model:

```bash
python -m spacy download en_core_web_sm
```

---

## 🚀 Running the Chatbot

From the project root:

```bash
python -m app.chatbot_text
```

To exit:

```
quit
```

---

## 🧪 LLM Fallback – Teacher Mode

* Activated only when intent + memory + knowledge fail
* Generates a response using an LLM
* Question + answer are stored in `data/llm_memory.json`
* These examples are later converted into training data

Environment variable required:

```bash
setx OPENAI_API_KEY "your_api_key_here"
```

The LLM **teaches the bot**, then steps back.

---

## 📚 Learning Pipeline (After Conversations)

After chatting with the bot, improve it using:

```bash
python training/discover_intents.py
python training/auto_append_intents.py
python training/auto_generate_responses.py
python training/train_chatbot.py
```

### What happens in this pipeline

* Unrecognized queries are clustered
* New intents are created automatically
* Safe, neutral responses are generated
* Intent model is retrained
* Future LLM usage decreases

---

## 🧠 Semantic Memory

* Uses **FAISS + sentence‑transformers**
* Fully offline after installation
* Persistent across restarts
* Importance‑weighted storage
* Automatic forgetting of low‑value memories

This allows the chatbot to remember **concepts**, not just exact phrases.

---

## 🔐 Files Not Committed to Git

Runtime and personal data are excluded for safety:

* chat history
* unrecognized queries
* LLM memory logs
* FAISS index files
* trained model files

This keeps the repository clean and safe to share.

---

## 📌 Notes

* Designed for **learning and experimentation**
* Follows real‑world AI system patterns
* Emphasizes safety, reproducibility, and clarity
* LLM improves the bot over time instead of replacing it

---

## 📄 License

MIT License
