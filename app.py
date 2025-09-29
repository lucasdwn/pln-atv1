from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ------------------------
# Config - modelo local
# ------------------------
MODEL_NAME = "google/flan-t5-large"

print("🔄 Carregando modelo...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Modelo de embeddings (para busca semântica)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dim = 384
index = faiss.IndexFlatIP(dim)  # inner product
documents = []

# ------------------------
# FastAPI
# ------------------------
app = FastAPI()

class Ingest(BaseModel):
    text: str

class Ask(BaseModel):
    question: str

@app.post("/ingest")
def ingest(item: Ingest):
    vec = embedder.encode([item.text], convert_to_numpy=True, normalize_embeddings=True)
    index.add(vec)
    documents.append(item.text)
    return {"status": "added", "text": item.text}

@app.post("/ask")
def ask(item: Ask):
    # Embedding da pergunta
    q_vec = embedder.encode([item.question], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_vec, 1)
    context = documents[I[0][0]] if len(documents) > 0 else "Sem contexto disponível."

    # Prompt estruturado
    prompt = f"""Use o contexto abaixo para responder à pergunta de forma clara e direta.

Contexto:
{context}

Pergunta:
{item.question}

Resposta:"""

    # Tokenização + geração
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9
    )

    # Decodificação + limpeza
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = decoded.strip()

    return {"answer": answer, "context": context}

@app.get("/health")
def health():
    return {"status": "ok"}
