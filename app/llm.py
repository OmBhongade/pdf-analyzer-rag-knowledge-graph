import requests
from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL

def generate_answer(question: str, retrieved_docs: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {d['source']}, page {d['page']}]\n{d['text']}"
        for d in retrieved_docs
    )
    prompt = f"""You are PDF Analyzer, a careful document assistant.
Answer the question using ONLY the supplied context. If the answer is not
in the context, say that the uploaded documents do not contain enough
information. Do not invent facts. Cite sources using [filename, page N].

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=180
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()
