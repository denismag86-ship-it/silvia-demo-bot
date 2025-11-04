from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from bs4 import BeautifulSoup
import re
import time
import hashlib
import os
import logging
from pydantic import BaseModel
from openai import AsyncOpenAI
import chromadb
from chromadb.config import Settings

# --- Логирование ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Конфигурация ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Требуется переменная окружения OPENAI_API_KEY")

COLLECTION_NAME = "demo_sites"
SESSION_TTL_SECONDS = 3600

# --- Инициализация FastAPI ---
app = FastAPI(title="Silvia API", version="1.0.0")

# 🔥 CORS — ИСПРАВЛЕНО: убраны пробелы
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://silvia-ai.ru",
        "https://www.silvia-ai.ru",
        "http://localhost:8000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# 🔥 ChromaDB — ИСПРАВЛЕНО: используем in-memory для Render
try:
    # In-memory режим для эфемерного хранилища Render
    # Отключаем телеметрию чтобы убрать ошибки в логах
    chroma_client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        chroma_telemetry_impl="none"  # Отключаем телеметрию полностью
    ))
    logger.info("✅ ChromaDB initialized (in-memory mode)")
except Exception as e:
    logger.error(f"❌ ChromaDB initialization failed: {e}")
    chroma_client = None

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

def extract_main_content(html: str, url: str):
    """Извлекает только основной контент сайта, удаляя шум."""
    soup = BeautifulSoup(html, "lxml")
    
    # Удаляем всё лишнее
    for tag in soup(["script", "style", "nav", "footer", "aside", "header", "form", "button", "img", "svg", "noscript"]):
        tag.decompose()
    
    # Ищем основной контент
    main = soup.find("main") or soup.find("article") or soup.find("section") or soup.body
    if main:
        text = main.get_text(separator=" ", strip=True)
    else:
        text = soup.get_text(separator=" ", strip=True)
    
    # Очищаем пробелы
    text = re.sub(r"\s+", " ", text).strip()
    
    # Получаем название компании
    title = soup.title.string if soup.title else ""
    company_name = title or url.split("//")[-1].split("/")[0]
    lang = soup.html.get("lang", "ru") if soup.html else "ru"
    
    return {"text": text, "company_name": company_name, "lang": lang}

def smart_truncate(text: str, max_chars: int = 2800) -> str:
    """Обрезает текст до последнего полного предложения."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_end = max(
        truncated.rfind(". "),
        truncated.rfind("! "),
        truncated.rfind("? "),
        truncated.rfind(".\n"),
    )
    if last_end != -1:
        return truncated[:last_end + 1]
    return truncated[:max_chars]

# --- Эндпоинты ---
@app.get("/")
@app.head("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Silvia API",
        "version": "1.0.0",
        "endpoints": ["/analyze", "/chat", "/health"]
    }

@app.get("/health")
@app.head("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "chromadb": "connected" if chroma_client else "disconnected",
        "openai": "configured" if OPENAI_API_KEY else "missing"
    }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    if not chroma_client:
        raise HTTPException(status_code=503, detail="ChromaDB not available")
    
    url = req.url.strip()
    logger.info(f"📊 Analyzing URL: {url}")
    
    if not is_valid_url(url):
        raise HTTPException(status_code=400, detail="Invalid URL")
    
    session_id = generate_session_id(url)
    
    try:
        # Получаем HTML
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as http_client:
            resp = await http_client.get(url)
            resp.raise_for_status()
            html = resp.text
        
        logger.info(f"✅ HTML fetched: {len(html)} chars")
        
        # Извлекаем контент
        data = extract_main_content(html, url)
        raw_text = data["text"]
        
        if not raw_text:
            raise HTTPException(status_code=400, detail="No meaningful content found on the site")
        
        logger.info(f"📝 Extracted text: {len(raw_text)} chars")
        
        # Обрезаем до безопасного размера
        safe_text = smart_truncate(raw_text, max_chars=2800)
        logger.info(f"✂️ Truncated to: {len(safe_text)} chars")
        
        # Генерируем эмбеддинг
        embedding_resp = await client.embeddings.create(
            input=safe_text, 
            model="text-embedding-3-small"
        )
        embedding = embedding_resp.data[0].embedding
        logger.info(f"🧠 Embedding created: {len(embedding)} dimensions")
        
        # Сохраняем в ChromaDB
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        collection.upsert(
            ids=[session_id],
            embeddings=[embedding],
            documents=[safe_text],
            metadatas=[{
                "url": url,
                "company_name": data["company_name"],
                "lang": data["lang"],
                "created_at": int(time.time())
            }]
        )
        
        logger.info(f"✅ Session created: {session_id}")
        return AnalyzeResponse(session_id=session_id)
    
    except httpx.HTTPError as e:
        logger.error(f"❌ HTTP error: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not chroma_client:
        raise HTTPException(status_code=503, detail="ChromaDB not available")
    
    session_id = req.session_id
    question = req.question.strip()
    
    logger.info(f"💬 Chat request: session={session_id}, question={question[:50]}...")
    
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")
    
    try:
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
            welcome = f"Hi! I'm the AI assistant for {company_name}. How can I help you today?"
        else:
            welcome = f"Здравствуйте! Я — цифровой помощник компании {company_name}. Чем могу помочь?"

        if len(question) < 5 and any(w in question.lower() for w in ["прив", "hi", "hello", "здрав"]):
            return ChatResponse(answer=welcome)

        # Системный промт
        system_prompt = f"""Вы — Silvia, интеллектуальный цифровой сотрудник компании «{company_name}». 
Ваша задача — отвечать от лица компании, используя ТОЛЬКО информацию с её главной страницы.

Правила:
1. Говорите дружелюбно, профессионально и с лёгкой креативностью.
2. НЕ выдумывайте факты. Если информации нет — скажите: «Это не указано на сайте, но я могу уточнить у команды!»
3. Избегайте фраз вроде «На сайте написано…». Вы — голос компании.
4. Ответы — краткие (1–3 предложения), но полезные.
5. Если вопрос не по теме — мягко верните в контекст.

Контекст (не цитируйте дословно):
{document}
"""

        chat_resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.75,
            max_tokens=300,
            top_p=0.9
        )
        answer = chat_resp.choices[0].message.content.strip()
        logger.info(f"✅ Answer generated: {len(answer)} chars")
        return ChatResponse(answer=answer)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
