# Полное отключение телеметрии chromadb через monkey patching
import os
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_DISABLE_OPENTELEMETRY"] = "true"
os.environ["CHROMA_DISABLE_EVENTS"] = "true"

# Импорты
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from bs4 import BeautifulSoup
import re
import time
import logging
import hashlib
from pydantic import BaseModel
from openai import AsyncOpenAI
import chromadb
from chromadb.config import Settings

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# --- Конфигурация ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Требуется переменная окружения OPENAI_API_KEY")

COLLECTION_NAME = "demo_sites"
SESSION_TTL_SECONDS = 3600

# --- Полное отключение телеметрии chromadb ---
# Monkey patching для полного отключения телеметрии
def mock_capture(*args, **kwargs):
    return None

try:
    # Патчим PostHog клиент для полного отключения телеметрии
    import chromadb.telemetry.product.posthog
    chromadb.telemetry.product.posthog.Posthog = type('MockPosthog', (), {
        'capture': staticmethod(mock_capture),
        '_capture': staticmethod(mock_capture),
        '__init__': lambda *args, **kwargs: None
    })
except Exception as e:
    logger.warning(f"Не удалось полностью отключить телеметрию: {str(e)}")

# --- Инициализация FastAPI ---
app = FastAPI()

# 🔥 CORS — исправлено: убраны лишние пробелы в URL
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

# Инициализация ChromaDB с полным отключением телеметрии
chroma_client = chromadb.Client(Settings(
    anonymized_telemetry=False,
    allow_reset=False,
    is_persistent=False
))

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

# --- Кэш коллекции ---
_collection_cache = None

def get_collection():
    global _collection_cache
    if _collection_cache is None:
        try:
            logger.info(f"🔍 Получаем коллекцию: {COLLECTION_NAME}")
            _collection_cache = chroma_client.get_collection(name=COLLECTION_NAME)
        except:
            logger.info(f"🆕 Создаем новую коллекцию: {COLLECTION_NAME}")
            _collection_cache = chroma_client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
    return _collection_cache

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
    logger.info("🧹 Извлечение основного контента...")
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
    
    logger.info(f"📝 Извлечено {len(text)} символов")
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
    return truncated[:max_chars]  # fallback

# --- Эндпоинты ---
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    url = req.url.strip()
    logger.info(f"📊 Анализ URL: {url}")
    
    if not is_valid_url(url):
        logger.warning(f"URLException: Неверный URL - {url}")
        raise HTTPException(status_code=400, detail="Invalid URL")
    
    session_id = generate_session_id(url)
    logger.info(f"🆔 Сгенерирован session_id: {session_id}")
    
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as http_client:
            logger.info(f"🌐 Запрос к {url}")
            resp = await http_client.get(url)
            resp.raise_for_status()
            html = resp.text
            logger.info(f"✅ HTML загружен: {len(html)} символов")
        
        data = extract_main_content(html, url)
        raw_text = data["text"]
        
        if not raw_text or len(raw_text) < 50:
            logger.warning(f"_ContentWarning: Недостаточно контента на сайте {url}")
            raise HTTPException(status_code=400, detail="No meaningful content found on the site")
        
        # Обрезаем до безопасного размера
        safe_text = smart_truncate(raw_text, max_chars=2800)
        logger.info(f"✂️ Текст обрезан до: {len(safe_text)} символов")
        
        # Генерируем эмбеддинг
        logger.info("🧠 Генерация эмбеддинга...")
        embedding_resp = await client.embeddings.create(input=safe_text, model="text-embedding-3-small")
        embedding = embedding_resp.data[0].embedding
        logger.info(f"✅ Эмбеддинг создан: {len(embedding)} измерений")
        
        # Работа с коллекцией
        collection = get_collection()
        logger.info("💾 Сохранение в коллекцию...")
        
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
        logger.info(f"✅ Сессия создана: {session_id}")
        return AnalyzeResponse(session_id=session_id)
    
    except Exception as e:
        error_detail = str(e)
        logger.error(f"❌ Ошибка при анализе: {error_detail}")
        
        # Специальная обработка ошибок сети
        if "timeout" in error_detail.lower() or "connect" in error_detail.lower():
            error_detail = "Сайт не отвечает или слишком медленно загружается. Попробуйте позже."
        elif "status code 4" in error_detail.lower() or "status code 5" in error_detail.lower():
            error_detail = "Не удалось получить доступ к сайту. Убедитесь, что сайт доступен из интернета."
        
        raise HTTPException(status_code=500, detail=f"Ошибка анализа сайта: {error_detail}")

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id
    question = req.question.strip()
    logger.info(f"💬 Чат-запрос для сессии {session_id}: '{question[:50]}...'")
    
    if not question:
        logger.warning("⚠️ Пустой вопрос")
        raise HTTPException(status_code=400, detail="Question is empty")
    
    try:
        collection = get_collection()
        results = collection.get(ids=[session_id], include=["documents", "metadatas"])
    except Exception as e:
        logger.error(f"❌ Ошибка доступа к коллекции: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка базы данных")
    
    if not results["ids"]:
        logger.warning(f"⚠️ Сессия не найдена: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
    
    created_at = results["metadatas"][0]["created_at"]
    if time.time() - created_at > SESSION_TTL_SECONDS:
        collection.delete(ids=[session_id])
        logger.info(f"🧹 Удалена устаревшая сессия: {session_id}")
        raise HTTPException(status_code=410, detail="Session expired")
    
    document = results["documents"][0]
    company_name = results["metadatas"][0]["company_name"]
    lang = results["metadatas"][0]["lang"]
    logger.info(f"🏢 Компания: {company_name}, Язык: {lang}")

    # Приветствие
    if lang == "en":
        welcome = f"Hi! I'm the AI assistant for {company_name}. How can I help you today?"
    else:
        welcome = f"Здравствуйте! Я — цифровой помощник компании {company_name}. Чем могу помочь?"

    if len(question) < 5 and any(w in question.lower() for w in ["прив", "hi", "hello", "здрав", "здар", "привет", "ку"]):
        logger.info("👋 Обнаружено приветствие")
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

    try:
        logger.info("🤖 Генерация ответа...")
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
        logger.info(f"✅ Ответ сгенерирован: '{answer[:50]}...'")
        return ChatResponse(answer=answer)
    except Exception as e:
        error_detail = str(e)
        logger.error(f"❌ Ошибка генерации: {error_detail}")
        
        if "APIConnectionError" in error_detail:
            error_detail = "Сервис временно недоступен. Попробуйте позже."
        elif "AuthenticationError" in error_detail:
            error_detail = "Ошибка аутентификации сервиса. Обратитесь к администратору."
        
        raise HTTPException(status_code=500, detail=f"Ошибка генерации ответа: {error_detail}")

# --- Health check endpoint ---
@app.get("/health")
async def health_check():
    logger.info("❤️ Health check запрошен")
    chroma_status = "ok"
    try:
        collection = get_collection()
        chroma_status = f"ok (count: {collection.count()})"
    except Exception as e:
        chroma_status = f"error: {str(e)}"
    
    return {
        "status": "ok", 
        "service": "silvia-ai-demo",
        "chroma_db": chroma_status,
        "openai_api_key_present": bool(OPENAI_API_KEY),
        "timestamp": int(time.time())
    }

@app.get("/")
async def root():
    return {"message": "Silvia AI Demo API", "version": "1.0"}

