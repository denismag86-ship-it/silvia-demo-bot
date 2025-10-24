from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from bs4 import BeautifulSoup
import re
import time
import hashlib
import os
from pydantic import BaseModel
from openai import AsyncOpenAI
import chromadb
from chromadb.config import Settings

# --- Конфигурация ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Требуется переменная окружения OPENAI_API_KEY")

COLLECTION_NAME = "demo_sites"
SESSION_TTL_SECONDS = 3600

# --- Инициализация FastAPI ---
app = FastAPI()

# 🔥 CORS — исправлено: убраны пробелы
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://silvia-ai.ru",
        "https://www.silvia-ai.ru",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

# --- Модели ---
class AnalyzeRequest(BaseModel):
    url: str

class AnalyzeResponse(BaseModel):
    session_id: str

class ChatRequest(BaseModel):
    session_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str

# --- Вспомогательные функции ---
def is_valid_url(url: str) -> bool:
    try:
        result = httpx.URL(url)
        return result.scheme in ("http", "https") and bool(result.host)
    except Exception:
        return False

def generate_session_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]

def extract_text_and_metadata(html: str, url: str):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
        tag.decompose()
    title = soup.title.string if soup.title else ""
    company_name = title or url.split("//")[-1].split("/")[0]
    lang = soup.html.get("lang", "ru") if soup.html else "ru"
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return {"text": text, "company_name": company_name, "lang": lang}

def smart_truncate(text: str, max_chars: int = 2800) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_end = max(truncated.rfind(". "), truncated.rfind("! "), truncated.rfind("? "))
    if last_end != -1:
        return truncated[:last_end + 1]
    return truncated

# --- Эндпоинты ---
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    url = req.url.strip()
    if not is_valid_url(url):
        raise HTTPException(status_code=400, detail="Invalid URL")
    session_id = generate_session_id(url)
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as http_client:
            resp = await http_client.get(url)
            resp.raise_for_status()
            html = resp.text
        data = extract_text_and_metadata(html, url)
        text = data["text"]
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found on site")
        embedding_resp = await client.embeddings.create(input=text, model="text-embedding-3-small")
        embedding = embedding_resp.data[0].embedding
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        collection.upsert(
            ids=[session_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "url": url,
                "company_name": data["company_name"],
                "lang": data["lang"],
                "created_at": int(time.time())
            }]
        )
        return AnalyzeResponse(session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    results = collection.get(ids=[session_id])
    if not results["ids"]:
        raise HTTPException(status_code=404, detail="Session not found")
    created_at = results["metadatas"][0]["created_at"]
    if time.time() - created_at > SESSION_TTL_SECONDS:
        collection.delete(ids=[session_id])
        raise HTTPException(status_code=410, detail="Session expired")
    document = results["documents"][0]
    company_name = results["metadatas"][0]["company_name"]
    lang = results["metadatas"][0]["lang"]

    # Приветствие
    if lang == "en":
        welcome = f"Hi! I’m the AI assistant for {company_name}. How can I help you today?"
    else:
        welcome = f"Здравствуйте! Я — цифровой помощник компании {company_name}. Чем могу помочь?"

    if len(question) < 5 and any(w in question.lower() for w in ["прив", "hi", "hello", "здрав"]):
        return ChatResponse(answer=welcome)

    # Умная обрезка контекста
    context = smart_truncate(document)

    # 🧠 УЛУЧШЕННЫЙ СИСТЕМНЫЙ ПРОМТ
    system_prompt = f"""Вы — Silvia, интеллектуальный цифровой сотрудник компании «{company_name}». 
Ваша задача — отвечать от лица компании, используя ТОЛЬКО информацию с её главной страницы.

Правила:
1. Говорите дружелюбно, профессионально и с лёгкой креативностью: используйте метафоры, задавайте уточняющие вопросы, подчёркивайте выгоды.
2. НЕ выдумывайте факты. Если информации нет — скажите: «Это не указано на сайте, но я могу уточнить у команды!»
3. Избегайте фраз вроде «На сайте написано…». Вы — голос компании, а не парсер.
4. Делайте ответы краткими (1–3 предложения), но содержательными.
5. Если вопрос не по теме — мягко возвращайте в контекст: «Я помогаю с вопросами о {company_name}. Например, вы можете спросить о наших услугах или как с нами связаться.»

Контекст с сайта (используйте его осмысленно, не цитируйте дословно):
{context}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    try:
        chat_resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.75,      # ↑ креативность
            max_tokens=300,
            top_p=0.9
        )
        answer = chat_resp.choices[0].message.content.strip()
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
