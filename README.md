# RAG Tourism Chatbot API

## 🚀 Chạy Demo

### 1. Cài đặt dependencies

```powershell
pip install -r requirements.txt
```

### 2. Cấu hình file .env

Tạo file `.env` với nội dung:

```env
GROQ_API_KEY=your-groq-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=rag-tourism
RAG_TOP_K=5
LLM_MODEL=llama-3.3-70b-versatile
PORT=8080
```

### 3. Import dữ liệu (chỉ chạy lần đầu)

```powershell
python store_data.py
```

### 4. Chạy server

```powershell
uvicorn main:app --reload --port 8080
```

### 5. Test API

```powershell
# Health check
Invoke-RestMethod -Uri http://localhost:8080/health

# Ask question
$body = @{ query = "Du lịch Hà Nội có gì hay?" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8080/ask -ContentType 'application/json' -Body $body
```

**Swagger UI:** http://localhost:8080/docs
