"""
Chatbot (Python version) - ICT project.

1. RETRIEVE: finds the best answer in data/knowledge.txt using TF-IDF + cosine similarity
   (implemented from scratch, no libraries).
2. GENERATE (optional): if a local LLM is running with Ollama, the retrieved facts are sent
   to the LLM as context (RAG) so it writes a natural answer.

   Offline mode (default):  python chatbot.py
   LLM mode:                LLM_BACKEND=ollama LLM_MODEL=llama3.2 python chatbot.py
"""
import json
import math
import os
import re
import urllib.request

KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "knowledge.txt")
STOP = {"the", "a", "an", "is", "are", "of", "to", "in", "and", "what", "how", "do", "does",
        "i", "me", "you", "it", "for", "on", "about", "tell", "explain"}


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9\+]+", text.lower()) if w not in STOP]


class Retriever:
    def __init__(self, path):
        self.entries = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if "|" in line:
                    q, a = line.strip().split("|", 1)
                    self.entries.append((q, a))
        self.docs = [tokenize(q + " " + a) for q, a in self.entries]
        n = len(self.docs)
        vocab = {w for d in self.docs for w in d}
        self.idf = {w: math.log((1 + n) / (1 + sum(w in d for d in self.docs))) + 1 for w in vocab}
        self.vectors = [self._vector(d) for d in self.docs]

    def _vector(self, tokens):
        vec = {}
        for w in tokens:
            if w in self.idf:
                vec[w] = vec.get(w, 0) + self.idf[w]
        return vec

    @staticmethod
    def _cosine(a, b):
        dot = sum(a[w] * b.get(w, 0) for w in a)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def search(self, query, k=2):
        qv = self._vector(tokenize(query))
        scored = sorted(((self._cosine(qv, v), i) for i, v in enumerate(self.vectors)), reverse=True)
        return [(s, self.entries[i]) for s, i in scored[:k] if s > 0.12]


def ask_llm(question, context, history):
    """Call a local Ollama server. Returns None if unavailable."""
    prompt = ("Answer the question using the facts below. Be short and clear.\n\nFacts:\n"
              + "\n".join(context) + "\n\nConversation so far:\n" + "\n".join(history[-4:])
              + f"\n\nQuestion: {question}\nAnswer:")
    body = json.dumps({"model": os.getenv("LLM_MODEL", "llama3.2"), "prompt": prompt, "stream": False}).encode()
    try:
        req = urllib.request.Request("http://localhost:11434/api/generate", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["response"].strip()
    except Exception:
        return None


def reply(retriever, question, history, use_llm):
    hits = retriever.search(question)
    if not hits:
        return "Sorry, I don't know that yet. Try asking about AI, Python, C++, Git or hackathons."
    if use_llm:
        answer = ask_llm(question, [a for _, (_, a) in hits], history)
        if answer:
            return answer
    return hits[0][1][1]


def main():
    retriever = Retriever(KB_PATH)
    use_llm = os.getenv("LLM_BACKEND") == "ollama"
    history = []
    print(f"Chatbot ready ({'LLM + RAG' if use_llm else 'offline retrieval'} mode). Type 'quit' to exit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if not q:
            continue
        a = reply(retriever, q, history, use_llm)
        history += [f"User: {q}", f"Bot: {a}"]
        print("Bot:", a)


if __name__ == "__main__":
    main()
