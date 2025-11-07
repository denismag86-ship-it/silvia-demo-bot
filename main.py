import os
import re
import time
import logging
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI

# --- Логирование ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("silvia")

# --- Конфигурация ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Требуется переменная окружения OPENAI_API_KEY")

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
ALLOW_JINA_FALLBACK = os.getenv("ALLOW_JINA_FALLBACK", "1") == "1"

# 🔧 ИСПРАВЛЕНИЕ 1: Добавьте все возможные варианты вашего фронтенда
ALLOWED_ORIGINS = [
    "https://silvia-ai.ru",
    "https://www.silvia-ai.ru",
    "http://silvia-ai.ru",
    "http://www.silvia-ai.ru",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:3000",
]

# --- Клиенты ---
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- Модели ---
class AnalyzeRequest(BaseModel):
    url: str

class AnalyzeResponse(BaseModel):
    url: str
    document: str
    company_name: str
    lang: str

class ChatRequest(BaseModel):
    question: str
    document: str
    company_name: Optional[str] = None
    lang: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str

# --- Инициализация FastAPI ---
app = FastAPI(title="Silvia API (stateless)", version="2.0.1")

# 🔧 ИСПРАВЛЕНИЕ 2: Временно разрешите все origins для теста
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ ДЛЯ ТЕСТА! Потом верните ALLOWED_ORIGINS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 🔧 ИСПРАВЛЕНИЕ 3: Middleware для логирования ВСЕХ запросов
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    
    # Логируем входящий запрос
    logger.info(f"📨 {request.method} {request.url.path} от {request.client.host}")
    logger.info(f"   Headers: Origin={request.headers.get('origin')}, Content-Type={request.headers.get('content-type')}")
    
    try:
        response = await call_next(request)
        duration = time.time() - start
        
        logger.info(f"✅ {request.method} {request.url.path} → {response.status_code} ({duration:.2f}s)")
        return response
        
    except Exception as e:
        duration = time.time() - start
        logger.error(f"❌ {request.method} {request.url.path} → ERROR ({duration:.2f}s): {e}")
        raise

# --- Утилиты ---
def normalize_url(url: str) -> str:
    u = url.strip()
    if not re.match(r"^https?://", u, flags=re.I):
        u = "https://" + u
    return u

def is_valid_url(url: str) -> bool:
    try:
        parsed = httpx.URL(url)
        return parsed.scheme in ("http", "https") and bool(parsed.host)
    except Exception:
        return False

def extract_main_content(html: str, url: str):
    """Извлечение основного контента из HTML"""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "aside", "header", "form", "button", "img", "svg", "noscript"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.find("section") or soup.find("div", class_=re.compile(r"content|main", re.I))
    if not main and soup.body:
        main = soup.body
    
    text = (main or soup).get_text(separator=" ", strip=True) if soup else ""
    text = re.sub(r"\s+", " ", text).strip()

    title = ""
    if soup and soup.title and soup.title.string:
        title = soup.title.string.strip()
    
    company_name = title if title else url.split("//")[-1].split("/")[0]

    lang = "ru"
    if soup and soup.html:
        html_lang = soup.html.get("lang")
        if html_lang:
            lang = html_lang.lower().split("-")[0]

    return {"text": text, "company_name": company_name, "lang": lang}

def smart_truncate(text: str, max_chars: int = 2800) -> str:
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

UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

async def fetch_html_best_effort(url: str) -> tuple[str, str]:
    async with httpx.AsyncClient(
        timeout=20.0,  # Сокращено для быстрого ответа
        follow_redirects=True,
        verify=False
    ) as http_client:
        
        # Только 1 попытка для скорости
        headers = {
            "User-Agent": UA_LIST[0],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
        }
        
        try:
            logger.info(f"🌐 Загрузка {url}...")
            r = await http_client.get(url, headers=headers)
            
            if r.status_code < 400 and r.text.strip():
                logger.info(f"✅ Загружено {len(r.text)} символов")
                return r.text, str(r.url)
            
            logger.warning(f"⚠️ Статус {r.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Ошибка: {e}")

        # HTTP fallback
        if url.startswith("https://"):
            alt_url = url.replace("https://", "http://", 1)
            try:
                r = await http_client.get(alt_url, headers=headers, timeout=15.0)
                if r.status_code < 400 and r.text.strip():
                    return r.text, alt_url
            except:
                pass

        # Jina AI fallback
        if ALLOW_JINA_FALLBACK:
            try:
                jina_url = f"https://r.jina.ai/{url}"
                jr = await http_client.get(jina_url, headers=headers, timeout=20.0)
                
                if jr.status_code < 400 and jr.text.strip():
                    safe_text = jr.text.replace("<", "&lt;").replace(">", "&gt;")
                    html = f"<html><head><title>Content</title></head><body><main>{safe_text}</main></body></html>"
                    return html, url
            except:
                pass

    raise HTTPException(status_code=503, detail="Не удалось загрузить сайт")

# --- Эндпоинты ---
@app.get("/")
@app.head("/")
async def root():
    return {
        "status": "ok", 
        "service": "Silvia API", 
        "version": "2.0.1",
        "time": int(time.time()),
    }

@app.get("/health")
@app.head("/health")
async def health():
    return {
        "status": "healthy",
        "openai": "configured" if OPENAI_API_KEY else "missing",
        "time": int(time.time()),
    }

# 🔧 ИСПРАВЛЕНИЕ 4: Тестовый эндпоинт
@app.post("/test-chat")
async def test_chat(request: Request):
    """Тестовый эндпоинт для проверки, доходят ли POST запросы"""
    try:
        body = await request.json()
        logger.info(f"🧪 TEST-CHAT получил: {body}")
        return {
            "status": "ok",
            "received": body,
            "message": "Backend работает! Проблема в frontend."
        }
    except Exception as e:
        logger.error(f"❌ TEST-CHAT error: {e}")
        return {"status": "error", "detail": str(e)}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    raw_url = req.url.strip()
    
    logger.info(f"📊 ANALYZE START: '{raw_url}'")
    
    if not raw_url:
        raise HTTPException(status_code=400, detail="URL не может быть пустым")
    
    url = normalize_url(raw_url)

    if not is_valid_url(url):
        raise HTTPException(status_code=400, detail="Некорректный URL")

    try:
        html, final_url = await fetch_html_best_effort(url)
        
        if not html or len(html) < 100:
            raise HTTPException(status_code=400, detail="Получен пустой контент")
        
        data = extract_main_content(html, final_url)
        
        if not data["text"] or len(data["text"]) < 50:
            raise HTTPException(status_code=400, detail="Не удалось извлечь контент")

        document = smart_truncate(data["text"], max_chars=2800)
        
        logger.info(f"✅ ANALYZE OK: {len(document)} символов, компания: '{data['company_name']}'")
        
        return AnalyzeResponse(
            url=final_url,
            document=document,
            company_name=data["company_name"],
            lang=data["lang"],
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ ANALYZE ERROR: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Ошибка: {str(e)[:200]}")

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    question = (req.question or "").strip()
    document = (req.document or "").strip()
    company_name = (req.company_name or "вашей компании").strip()
    lang = (req.lang or "ru").strip().split("-")[0]

    # 🔧 ИСПРАВЛЕНИЕ 5: Детальное логирование
    logger.info(f"💬 CHAT START")
    logger.info(f"   Question: '{question[:100]}'")
    logger.info(f"   Document length: {len(document)}")
    logger.info(f"   Company: '{company_name}'")
    logger.info(f"   Lang: '{lang}'")

    if not question:
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")
    
    if not document:
        raise HTTPException(status_code=400, detail="Документ пуст")

    # Проверка на приветствие
    q_lower = question.lower()
    greeting_words = ["привет", "здравствуй", "hi", "hello", "добрый"]
    
    if any(word in q_lower for word in greeting_words):
        if lang == "en":
            welcome = f"Hi! I'm Silvia, AI assistant for {company_name}. How can I help?"
        else:
            welcome = f"Здравствуйте! Я — Сильвия, помощник {company_name}. Чем могу помочь?"
        
        logger.info(f"✅ CHAT: приветствие отправлено")
        return ChatResponse(answer=welcome)

    # Системный промпт
    system_prompt = f"""Вы — Сильвия, помощник компании «{company_name}».
Отвечайте кратко (1-2 предложения) на основе контекста ниже.

Контекст:
{document[:2000]}
"""

    try:
        logger.info(f"🤖 Отправка в OpenAI...")
        
        chat_resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        
        answer = chat_resp.choices[0].message.content.strip() if chat_resp.choices else ""
        
        if not answer:
            answer = "Извините, не удалось сгенерировать ответ."
        
        logger.info(f"✅ CHAT OK: {len(answer)} символов")
        logger.info(f"   Answer: '{answer[:100]}'")
        
        return ChatResponse(answer=answer)

    except Exception as e:
        logger.error(f"❌ CHAT ERROR: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="OpenAI временно недоступен")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
