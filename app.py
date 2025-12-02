from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

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
# Pré-ingest dos dados DSM
# ------------------------
def load_dsm_data():
    """Carrega e indexa automaticamente o arquivo dsm_data.txt"""
    dsm_file = "dsm_data.txt"
    
    if not os.path.exists(dsm_file):
        print(f"⚠️  Arquivo {dsm_file} não encontrado. Pulando pré-ingest.")
        return
    
    print(f"📚 Carregando dados do arquivo {dsm_file}...")
    
    with open(dsm_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Dividir o conteúdo em chunks por semestre e disciplinas
    chunks = []
    
    # Separar por seções de semestre
    sections = content.split("=== ")
    for section in sections:
        if section.strip():
            # Adicionar a seção completa como um chunk
            section_text = section.strip()
            if len(section_text) > 50:  # Ignorar textos muito curtos
                chunks.append(section_text)
            
            # Também adicionar cada disciplina como um chunk separado
            lines = section.split("\n")
            current_discipline = ""
            for i, line in enumerate(lines):
                line = line.strip()
                # Detectar linhas que começam com código de disciplina
                if line and any(line.startswith(prefix) for prefix in ["IAL", "ISO", "IBD", "ISW", "IES", "IED", "ILP", "MAT"]):
                    current_discipline = line
                    # Pegar a descrição (próxima linha não vazia)
                    if i + 1 < len(lines):
                        description = lines[i + 1].strip()
                        if description:
                            chunks.append(f"{current_discipline}\n{description}")
    
    # Indexar todos os chunks
    for chunk in chunks:
        if chunk:
            vec = embedder.encode([chunk], convert_to_numpy=True, normalize_embeddings=True)
            index.add(vec)
            documents.append(chunk)
    
    print(f"✅ Pré-ingest concluído! {len(chunks)} documentos indexados.")

# Executar o pré-ingest na inicialização
load_dsm_data()

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
