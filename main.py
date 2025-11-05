import os
import re
import time
import json
import logging
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI

# --- Логирование ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("silvia")

# --- Конфигурация ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Требуется переменная окружения OPENAI_API_KEY")

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
ALLOW_JINA_FALLBACK = os.getenv("ALLOW_JINA_FALLBACK", "1") == "1"

ALLOWED_ORIGINS = [
    "https://silvia-ai.ru",
    "https://www.silvia-ai.ru",
    "http://localhost:8000",
    "http://localhost:3000",
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
app = FastAPI(title="Silvia API (stateless)", version="2.0.0")

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
        u = "https://" + u
    return u

def is_valid_url(url: str) -> bool:
    try:
        parsed = httpx.URL(url)
        return parsed.scheme in ("http", "https") and bool(parsed.host)
    except Exception:
        return False

def extract_main_content(html: str, url: str):
    # Фолбэк парсера
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

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

UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
]

async def fetch_html_best_effort(url: str) -> tuple[str, str]:
    """
    Возвращает (html, final_url). Несколько UA + https->http + r.jina.ai fallback (опционально).
    """
    async with httpx.AsyncClient(timeout=25.0, follow_redirects=True, http2=True) as http_client:
        # 1) Несколько UA
        for ua in UA_LIST:
            headers = {
                "User-Agent": ua,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "ru,en;q=0.9",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            }
            try:
                r = await http_client.get(url, headers=headers)
                if r.status_code < 400 and r.text.strip():
                    return r.text, url
                if r.status_code in (401, 403, 406, 429):
                    continue
            except Exception:
                continue

        # 2) http fallback
        if url.startswith("https://"):
            alt = "http://" + url[len("https://"):]
            for ua in UA_LIST:
                headers = {
                    "User-Agent": ua,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "ru,en;q=0.9",
                }
                try:
                    r = await http_client.get(alt, headers=headers)
                    if r.status_code < 400 and r.text.strip():
                        return r.text, alt
                except Exception:
                    continue

        # 3) r.jina.ai fallback (возвращает уже текст)
        if ALLOW_JINA_FALLBACK:
            try:
                from urllib.parse import urlparse
                u = urlparse(url)
                jina_url = f"https://r.jina.ai/http://{u.netloc}{u.path}{'?' + u.query if u.query else ''}"
                jr = await http_client.get(jina_url, headers={"User-Agent": UA_LIST[0]})
                if jr.status_code < 400 and jr.text.strip():
                    safe = jr.text.replace("<", "&lt;").replace(">", "&gt;")
                    html = f"<html><body><main>{safe}</main></body></html>"
                    return html, url
            except Exception:
                pass

    # Если не получилось — вернем 403 для понятного UX
    raise HTTPException(status_code=403, detail="Сайт отклонил запросы (403). Попробуйте другой URL или прокси.")

# --- Эндпоинты ---
@app.get("/")
@app.head("/")
async def root():
    return {"status": "ok", "service": "Silvia API (stateless)", "version": "2.0.0", "endpoints": ["/analyze", "/chat", "/health"]}

@app.get("/health")
@app.head("/health")
async def health():
    return {
        "status": "healthy",
        "openai": "configured" if OPENAI_API_KEY else "missing",
        "mode": "stateless",
        "time": int(time.time()),
    }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    raw_url = req.url.strip()
    url = normalize_url(raw_url)
    logger.info(f"📊 Analyzing URL: {url}")

    if not is_valid_url(url):
        raise HTTPException(status_code=400, detail="Invalid URL")

    try:
        html, final_url = await fetch_html_best_effort(url)
        data = extract_main_content(html, final_url)
        if not data["text"]:
            raise HTTPException(status_code=400, detail="No meaningful content found on the site")

        document = smart_truncate(data["text"], max_chars=2800)
        return AnalyzeResponse(
            url=final_url,
            document=document,
            company_name=data["company_name"],
            lang=data["lang"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail="Failed to fetch or parse the site")

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    question = (req.question or "").strip()
    document = (req.document or "").strip()
    company_name = (req.company_name or "вашей компании").strip()
    lang = (req.lang or "ru").strip().split("-")[0]

    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")
    if not document:
        raise HTTPException(status_code=400, detail="Document is empty. Вызовите /analyze и передайте документ сюда.")

    # Приветствие
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

    try:
        chat_resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.7,
            max_tokens=300,
            top_p=0.9,
        )
        answer = chat_resp.choices[0].message.content.strip() if chat_resp.choices else "Извините, не удалось сгенерировать ответ."
        return ChatResponse(answer=answer)

    except Exception as e:
        logger.error(f"❌ Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="LLM временно недоступен, повторите попытку")
