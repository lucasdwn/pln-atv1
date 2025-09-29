

## Como rodar o projeto: 


### Criar ambiente virtual
```
# Criar ambiente virtual
python3 -m venv .venv

# Ativar ambiente
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows (PowerShell)
```
### Instalar dependencias
```
pip install -r requirements.txt
```

### Rodar aplicação
```
uvicorn app:app --reload --port 8000
```

## Como utilizar o projeto:

### Realizar Ingest
- Request de exemplo em request.http.


### Fazer Pergunta
- Request de exemplo em request.http.


## Evidências

### Ingest
![Ingest](./prints/ingest.png)

### Question
![Question](./prints/question.png)