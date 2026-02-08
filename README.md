# 🤖 Hybrid AI Chatbot with Semantic Memory & LLM Learning

A production‑style chatbot that combines **intent classification**, **semantic vector memory**, and an **LLM fallback used as a teacher (not a replacement)**.

The system improves over time by learning from conversations, discovering new intents, and retraining safely with human‑reviewable steps.

---

## ✨ Key Features

- Intent‑based chatbot (fast, local, inexpensive)
- Semantic vector memory using FAISS (offline & persistent)
- Knowledge base lookup
- Hinglish → English auto‑translation
- LLM fallback **only when the bot fails**
- LLM answers saved and reused for training
- Automatic intent discovery from conversations
- Safe retraining pipeline (no blind auto‑learning)
- Memory importance scoring & controlled forgetting

This is a **hybrid AI architecture**, similar to how real assistants are built in production.

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

## 🧠 How the Chatbot Responds (Decision Order)

1. **Semantic vector memory** (FAISS)
2. **Knowledge base lookup**
3. **Intent classification model**
4. **LLM fallback (teacher mode)**

The LLM is **never always‑on**. It is only used when the bot cannot confidently respond.

---

## 🧪 LLM Fallback – Teacher Mode

- Activated only when intent + memory + knowledge fail
- Generates a response using an LLM
- Question + answer are stored in `data/llm_memory.json`
- These examples are later converted into training data

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

### What happens in this pipeline:

- Unrecognized queries are clustered
- New intents are created automatically
- Safe, neutral responses are generated
- Intent model is retrained
- Future LLM usage decreases

---

## 🧠 Semantic Memory

- Uses **FAISS + sentence‑transformers**
- Fully offline after installation
- Persistent across restarts
- Importance‑weighted storage
- Automatic forgetting of low‑value memories

This allows the chatbot to remember *concepts*, not just exact phrases.

---

## 🔐 Files Not Committed to Git

Runtime and personal data are excluded for safety:

- chat history
- unrecognized queries
- LLM memory logs
- FAISS index files
- trained model files

This keeps the repository clean and safe to share.

---

## 📌 Notes

- Designed for **learning and experimentation**
- Follows real‑world AI system patterns
- LLM improves the bot over time instead of replacing it

---

