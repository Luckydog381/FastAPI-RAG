```
# 🧠 FastAPI-RAG

A powerful and lightweight project integrating **FastAPI**, **pgvector**, and **LangChain** for vector-based document operations.

---

## 🔧 Prerequisites

### 📌 Install PostgreSQL with pgvector
- Follow the [pgvector installation guide](https://github.com/pgvector/pgvector) to install and configure it.
- Make sure the `pgvector` extension is enabled in your PostgreSQL database.

### 📌 Environment Variables
Create a `.env` file in the `app/` directory (based on `env.example`) with the following structure:

```env
GEMINI_API_KEY=your_api_key_here
EMBEDDING_MODEL=text-embedding-005
EMBEDDING_SIZE=768
LLM_MODEL=gemini-2.0-flash
PG_VECTOR_DB_NAME=vector_db
PG_VECTOR_DB_USERNAME=postgres
PG_VECTOR_DB_PASSWORD=12345
PG_VECTOR_DB_HOST=localhost
PG_VECTOR_DB_PORT=5432
```

---

## 📦 Installation

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the environment

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Start the FastAPI server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Access the API

- Navigate to: `http://localhost:8000`

### View interactive API documentation

- Swagger UI: `http://localhost:8000/docs`

---

## 📝 Notes

- Ensure your PostgreSQL database is running and configured with the pgvector extension.
- Replace `your_api_key_here` with your actual API key.

---