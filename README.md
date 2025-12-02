

# Chatbot RAG - Grade Curricular DSM

Sistema de chatbot inteligente com RAG (Retrieval-Augmented Generation) para responder perguntas sobre a grade curricular do curso de Desenvolvimento de Software Multiplataforma (DSM).

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Modelo de Linguagem (LLM)](#modelo-de-linguagem-llm)
- [Implementação do RAG](#implementação-do-rag)
- [Limitações](#limitações)
- [Como Rodar o Projeto](#como-rodar-o-projeto)
- [Como Utilizar](#como-utilizar)
- [Exemplo de Diálogo](#exemplo-de-diálogo)
- [Evidências](#evidências)

---

## 🎯 Sobre o Projeto

Este projeto implementa um chatbot baseado em RAG (Retrieval-Augmented Generation) que utiliza:
- **LLM local**: Google Flan-T5 Large para geração de respostas
- **Embeddings**: sentence-transformers para busca semântica
- **Banco vetorial**: FAISS para indexação e recuperação eficiente
- **API REST**: FastAPI para interface de comunicação

O sistema carrega automaticamente a base de conhecimento sobre a grade DSM e responde perguntas de forma contextualizada.

---

## 🤖 Modelo de Linguagem (LLM)

### Modelo Escolhido: **Google Flan-T5 Large**

**Por que Flan-T5?**

1. **Gratuito e Open Source**: Totalmente gratuito, sem custos de API
2. **Execução Local**: Roda localmente, garantindo privacidade dos dados
3. **Otimizado para Instruções**: Treinado especificamente para seguir instruções (instruction-tuned)
4. **Tamanho Balanceado**: 780M parâmetros - bom equilíbrio entre qualidade e performance
5. **Multilíngue**: Suporta português, ideal para conteúdo em PT-BR
6. **Baixa Latência**: Respostas rápidas mesmo em hardware comum

**Especificações Técnicas:**
- **Arquitetura**: Transformer encoder-decoder (T5)
- **Parâmetros**: 780 milhões
- **Contexto**: Até 512 tokens de entrada
- **Framework**: Hugging Face Transformers + PyTorch

---

## 🔍 Implementação do RAG

O sistema RAG (Retrieval-Augmented Generation) foi implementado em 3 etapas:

### 1. **Indexação (Pré-ingest Automático)**

Na inicialização, o sistema:
- Carrega o arquivo `dsm_data.txt` com a grade curricular
- Divide o conteúdo em chunks inteligentes:
  - Seções por semestre (1º, 2º, 3º)
  - Disciplinas individuais com descrições
- Gera embeddings usando `sentence-transformers/all-MiniLM-L6-v2`
- Indexa no FAISS (IndexFlatIP com inner product)

```python
# Exemplo simplificado
vec = embedder.encode([chunk], normalize_embeddings=True)
index.add(vec)
documents.append(chunk)
```

### 2. **Recuperação (Retrieval)**

Quando uma pergunta chega:
- Converte a pergunta em embedding vetorial
- Busca no FAISS o documento mais similar (k=1)
- Recupera o contexto relevante da base de conhecimento

```python
q_vec = embedder.encode([question], normalize_embeddings=True)
D, I = index.search(q_vec, 1)  # Top-1 similar
context = documents[I[0][0]]
```

### 3. **Geração (Augmented Generation)**

- Monta um prompt estruturado com contexto + pergunta
- Envia para o Flan-T5 Large
- Gera resposta contextualizada baseada no conhecimento recuperado

```python
prompt = f"""Use o contexto abaixo para responder à pergunta de forma clara e direta.

Contexto:
{context}

Pergunta:
{question}

Resposta:"""
```

**Parâmetros de Geração:**
- `max_new_tokens=200`: Respostas concisas
- `temperature=0.7`: Equilíbrio entre criatividade e precisão
- `top_p=0.9`: Nucleus sampling para qualidade

---

## ⚠️ Limitações

### 1. **Capacidade do Modelo**
- Flan-T5 Large tem 780M parâmetros (menor que GPT-3.5/GPT-4)
- Pode gerar respostas menos sofisticadas em perguntas complexas
- Limitado a 512 tokens de contexto

### 2. **Busca Vetorial**
- Recupera apenas 1 documento (k=1) por consulta
- Pode perder informações relevantes em outros chunks
- Não faz fusão de múltiplas fontes

### 3. **Base de Conhecimento**
- Limitada às informações no `dsm_data.txt`
- Não tem conhecimento além da grade curricular fornecida
- Não se atualiza automaticamente

### 4. **Performance**
- Primeira execução baixa modelos (~3GB) - pode demorar
- Requer pelo menos 4GB de RAM disponível
- CPU-only pode ser lento (GPU recomendada)

### 5. **Idioma**
- Apesar de multilíngue, pode misturar inglês/português ocasionalmente
- Melhor desempenho em inglês que português

### 6. **Escalabilidade**
- FAISS em memória - perde dados ao reiniciar
- Não persiste o índice vetorial
- Não suporta múltiplos usuários concorrentes eficientemente

---

## 🚀 Como Rodar o Projeto

### 1. Criar Ambiente Virtual

```powershell
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente
.venv\Scripts\activate    # Windows (PowerShell)
```

### 2. Instalar Dependências

```powershell
pip install -r requirements.txt
```

**Nota**: A primeira execução baixará os modelos (~3GB). Aguarde a conclusão.

### 3. Rodar Aplicação

```powershell
uvicorn app:app --reload --port 8000
```

Você verá as mensagens:
```
🔄 Carregando modelo...
📚 Carregando dados do arquivo dsm_data.txt...
✅ Pré-ingest concluído! X documentos indexados.
```

A API estará disponível em: `http://localhost:8000`

---

## 💡 Como Utilizar

### Fazer Perguntas (Endpoint /ask)

O sistema já vem com a base DSM pré-carregada. Basta fazer perguntas!

**Exemplo de requisição:**

```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "question": "Quais são as disciplinas do 1º semestre?"
}
```

**Mais exemplos estão disponíveis no arquivo `request.http`**

### Adicionar Novos Dados (Endpoint /ingest) - Opcional

Caso queira adicionar informações adicionais:

```http
POST http://localhost:8000/ingest
Content-Type: application/json

{
    "text": "Texto adicional sobre o curso DSM..."
}
```

### Verificar Status (Endpoint /health)

```http
GET http://localhost:8000/health
```

---

## 💬 Exemplo de Diálogo (5 Interações)

### **Interação 1**
**Usuário**: "Quais são as disciplinas do 1º semestre do curso DSM?"

**Chatbot**: "O 1º semestre do curso DSM possui 6 disciplinas: IAL010 – Algoritmos e Lógica de Programação, ISO011 – Sistemas Operacionais e Redes de Computadores, IBD014 – Modelagem de Banco de Dados, ISW028 – Desenvolvimento Web I, IES011 – Engenharia de Software I, e ISW031 – Design Digital."

---

### **Interação 2**
**Usuário**: "O que é ensinado em Algoritmos e Lógica de Programação?"

**Chatbot**: "IAL010 – Algoritmos e Lógica de Programação é a disciplina fundamental que ensina os conceitos básicos de algoritmos, estruturas de controle, variáveis e lógica de programação. É a base para o desenvolvimento de software no curso."

---

### **Interação 3**
**Usuário**: "Quais disciplinas de banco de dados tem no curso?"

**Chatbot**: "O curso DSM possui três disciplinas de banco de dados: IBD014 – Modelagem de Banco de Dados no 1º semestre, que ensina diagrama entidade-relacionamento e normalização; IBD015 – Banco de Dados Relacional no 2º semestre, focado em SQL e PostgreSQL; e IBD016 – Banco de Dados Não Relacional no 3º semestre, que estuda MongoDB e bancos NoSQL."

---

### **Interação 4**
**Usuário**: "O que é estudado em Desenvolvimento Web II?"

**Chatbot**: "ISW029 – Desenvolvimento Web II é uma disciplina do 2º semestre que avança no desenvolvimento web com frameworks JavaScript modernos como React, Vue.js ou Angular, além de trabalhar com APIs RESTful e arquitetura frontend."

---

### **Interação 5**
**Usuário**: "O que é MongoDB e em qual semestre é estudado?"

**Chatbot**: "MongoDB é um banco de dados NoSQL estudado na disciplina IBD016 – Banco de Dados Não Relacional, oferecida no 3º semestre do curso DSM. A disciplina aborda modelagem de dados não relacionais, escalabilidade e performance em bancos NoSQL como MongoDB, Redis e Cassandra."

---

## 📸 Evidências

### Pré-ingest Automático
O sistema carrega automaticamente a base de conhecimento DSM na inicialização.

![Ingest](./prints/ingest.png)

### Pergunta 1 - Disciplinas do 1º Semestre
![Question](./prints/question.png)

### Pergunta 2 - Algoritmos e Lógica de Programação
![Ask 2](./prints/ask2.png)

### Pergunta 3 - Disciplinas de Banco de Dados
![Ask 3](./prints/ask3.png)

### Pergunta 4 - Desenvolvimento Web II
![Ask 4](./prints/ask4.png)

### Pergunta 5 - MongoDB
![Ask 5](./prints/ask5.png)

---

## 📦 Tecnologias Utilizadas

- **Python 3.8+**
- **FastAPI**: Framework web assíncrono
- **Transformers (Hugging Face)**: LLM Flan-T5
- **Sentence-Transformers**: Embeddings semânticos
- **FAISS**: Busca vetorial eficiente
- **PyTorch**: Backend de deep learning
- **Uvicorn**: Servidor ASGI

---

## 👨‍💻 Autor

Projeto desenvolvido como atividade da disciplina de Processamento de Linguagem Natural (PLN).

**Repositório**: [lucasdwn/pln-atv1](https://github.com/lucasdwn/pln-atv1)