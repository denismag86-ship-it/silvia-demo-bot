import os
import re
import time
import json
import hashlib
import logging
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI
from upstash_redis.asyncio import Redis

# --- Логирование ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("silvia")

# --- Конфигурация ---
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Требуется переменная окружения OPENAI_API_KEY")

UPSTASH_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")
if not UPSTASH_URL or not UPSTASH_TOKEN:
    raise ValueError("Требуются UPSTASH_REDIS_REST_URL и UPSTASH_REDIS_REST_TOKEN")

ALLOWED_ORIGINS = [
    "https://silvia-ai.ru",
    "https://www.silvia-ai.ru",
    "http://localhost:8000",
    "http://localhost:3000",
]

# --- Клиенты ---
client = AsyncOpenAI(api_key=OPENAI_API_KEY)
redis = Redis(url=UPSTASH_URL, token=UPSTASH_TOKEN)

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

# --- Инициализация FastAPI ---
app = FastAPI(title="Silvia API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Утилиты ---
def normalize_url(url: str) -> str:
    u = url.strip()
    if not re.match(r"^https?://", u, flags=re.I):
        u = "https://" + u  # по умолчанию https
    return u

def is_valid_url(url: str) -> bool:
    try:
        parsed = httpx.URL(url)
        return parsed.scheme in ("http", "https") and bool(parsed.host)
    except Exception:
        return False

def generate_session_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]

def extract_main_content(html: str, url: str):
    soup = BeautifulSoup(html, "lxml")

    # Удаляем шум
    for tag in soup(["script", "style", "nav", "footer", "aside", "header", "form", "button", "img", "svg", "noscript"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.find("section") or (soup.body if soup else None)
    text = (main or soup).get_text(separator=" ", strip=True) if soup else ""
    text = re.sub(r"\s+", " ", text).strip()

    title = ""
    if soup and soup.title and soup.title.string:
        title = soup.title.string.strip()
    company_name = title or url.split("//")[-1].split("/")[0]

    lang = "ru"
    if soup and soup.html and soup.html.get("lang"):
        lang = soup.html.get("lang").lower()
    lang = lang.split("-")[0]  # en-US -> en

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

# --- Эндпоинты ---
@app.get("/")
@app.head("/")
async def root():
    return {
        "status": "ok",
        "service": "Silvia API",
        "version": "1.2.0",
        "endpoints": ["/analyze", "/chat", "/health"]
    }

@app.get("/health")
@app.head("/health")
async def health():
    redis_status = "disconnected"
    try:
        pong = await redis.ping()
        redis_status = f"connected: {pong}"
    except Exception as e:
        redis_status = f"error: {e}"

    return {
        "status": "healthy",
        "redis": redis_status,
        "openai": "configured" if OPENAI_API_KEY else "missing",
    }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    raw_url = req.url.strip()
    url = normalize_url(raw_url)
    logger.info(f"📊 Analyzing URL: {url}")

    if not is_valid_url(url):
        raise HTTPException(status_code=400, detail="Invalid URL")

    session_id = generate_session_id(url)
    session_key = f"sess:{session_id}"

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 SilviaBot/1.0 (+https://silvia-ai.ru)",
            "Accept-Language": "ru,en;q=0.9",
        }
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as http_client:
            try:
                resp = await http_client.get(url, headers=headers)
                resp.raise_for_status()
            except Exception:
                # fallback на http, если https не открылся
                if url.startswith("https://"):
                    url_http = "http://" + url[len("https://"):]
                    resp = await http_client.get(url_http, headers=headers)
                    resp.raise_for_status()
                    url = url_http
                else:
                    raise
            html = resp.text

        logger.info(f"✅ HTML fetched: {len(html)} chars")

        data = extract_main_content(html, url)
        raw_text = data["text"]
        if not raw_text:
            raise HTTPException(status_code=400, detail="No meaningful content found on the site")

        logger.info(f"📝 Extracted text: {len(raw_text)} chars")

        safe_text = smart_truncate(raw_text, max_chars=2800)
        logger.info(f"✂️ Truncated to: {len(safe_text)} chars")

        session_data = {
            "url": url,
            "company_name": data["company_name"],
            "lang": data["lang"],
            "document": safe_text,
            "created_at": int(time.time()),
        }

        # Сохраняем с TTL
        await redis.set(session_key, json.dumps(session_data), ex=SESSION_TTL_SECONDS)

        logger.info(f"✅ Session created: {session_id}")
        return AnalyzeResponse(session_id=session_id)

    except httpx.HTTPError as e:
        logger.error(f"❌ HTTP error: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = (req.session_id or "").strip()
    question = (req.question or "").strip()
    logger.info(f"💬 Chat request: session={session_id}, question={question[:80]}...")

    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")

    try:
        session_key = f"sess:{session_id}"
        payload_raw = await redis.get(session_key)
        if not payload_raw:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        payload = json.loads(payload_raw)
        document = payload.get("document", "")
        company_name = payload.get("company_name", "вашей компании")
        lang = payload.get("lang", "ru")

        # Простое приветствие
        q = question.lower()
        if any(w in q for w in ["привет", "здрав", "hi", "hello", "hey"]):
            if lang == "en":
                welcome = f"Hi! I'm the AI assistant for {company_name}. How can I help you today?"
            else:
                welcome = f"Здравствуйте! Я — цифровой помощник компании {company_name}. Чем могу помочь?"
            return ChatResponse(answer=welcome)

        system_prompt = f"""Вы — Silvia, интеллектуальный цифровой сотрудник компании «{company_name}».
Отвечайте ТОЛЬКО на основе информации с главной страницы компании.

Правила:
1) Тон: дружелюбно и профессионально.
2) Не выдумывайте фактов. Если данных нет — скажите: «Этого нет на сайте, но я могу уточнить у команды!».
3) Не говорите «На сайте написано…». Вы — голос компании.
4) Ответы краткие (1–3 предложения), но полезные.
5) Вопросы не по теме — мягко возвращайте к тематике компании.

Контекст (не цитируйте дословно):
{document}
"""

        chat_resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.75,
            max_tokens=300,
            top_p=0.9,
        )

        answer = chat_resp.choices[0].message.content.strip() if chat_resp.choices else "Извините, не удалось сгенерировать ответ."
        logger.info(f"✅ Answer generated: {len(answer)} chars")
        return ChatResponse(answer=answer)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
