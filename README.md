# 💬 LLM-Style Chatbot — Python & C++ (ICT Project)

The same chatbot written in **two languages** to compare them.

It uses **retrieval (TF-IDF + cosine similarity)** to find the best answer from a knowledge base,
and — in the Python version — can optionally pass the retrieved facts to a **local LLM (Ollama)**
to write a natural reply. This pattern is called **RAG (Retrieval-Augmented Generation)**.

| | Python | C++ |
|---|---|---|
| File | `python/chatbot.py` | `cpp/chatbot.cpp` |
| Retrieval | TF-IDF + cosine (from scratch) | TF-IDF + cosine (from scratch) |
| LLM mode | ✅ optional via Ollama | ❌ retrieval only |
| Speed | good | very fast |

## Run
```bash
# Python (offline)
cd python && python chatbot.py

# Python with a local LLM (install Ollama first, then: ollama pull llama3.2)
LLM_BACKEND=ollama LLM_MODEL=llama3.2 python chatbot.py

# C++
cd cpp && g++ -std=c++17 -O2 chatbot.cpp -o chatbot && ./chatbot ../data/knowledge.txt
```

## Add your own knowledge
Edit `data/knowledge.txt`. One entry per line: `question|answer`. Both versions read the same file.

## How it works
1. Tokenize text and remove stop words
2. Weight each word by TF-IDF (rare words matter more)
3. Compare the question to every entry using cosine similarity
4. Return the best match (or let the LLM rewrite it using the matches as context)

## Future work
Embeddings with sentence-transformers, conversation memory, a web UI with Streamlit.
