import os
import re
import time
import logging
from typing import Optional, List

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI

# --- 1. Логирование ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("silvia")

# --- 2. Конфигурация ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("⚠️ Переменная OPENAI_API_KEY не найдена! Чат работать не будет.")

ALLOW_JINA_FALLBACK = os.getenv("ALLOW_JINA_FALLBACK", "1") == "1"

# --- 3. Инициализация FastAPI ---
app = FastAPI(title="Silvia API", version="2.2.0")

# --- 4. Настройка CORS ---
origins = [
    "https://silvia-ai.ru",
    "https://www.silvia-ai.ru",
    "http://silvia-ai.ru",
    "http://www.silvia-ai.ru",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. Клиенты ---
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- 6. Модели данных (Pydantic) ---

class AnalyzeRequest(BaseModel):
    url: str

class AnalyzeResponse(BaseModel):
    url: str
    document: str
    company_name: str
    lang: str

# 👇 Новая модель для одного сообщения в истории
class Message(BaseModel):
    role: str     # "user" или "assistant"
    content: str

class ChatRequest(BaseModel):
    question: str
    document: str
    # 👇 Новый список истории. По умолчанию пустой.
    history: List[Message] = [] 
    company_name: Optional[str] = None
    lang: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str

# --- 7. Middleware для логирования ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    path = request.url.path
    method = request.method
    
    # Логируем только начало важных запросов, чтобы не засорять
    if path in ["/chat", "/analyze"]:
        logger.info(f"📨 {method} {path} от {request.client.host}")
    
    try:
        response = await call_next(request)
        if path in ["/chat", "/analyze"]:
            duration = time.time() - start
            logger.info(f"✅ {method} {path} → {response.status_code} ({duration:.2f}s)")
        return response
    except Exception as e:
        duration = time.time() - start
        logger.error(f"❌ {method} {path} → ERROR ({duration:.2f}s): {e}")
        raise

# --- 8. Утилиты ---
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
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "aside", "form", "noscript", "iframe", "svg"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile(r"content|main|body", re.I))
    if not main and soup.body:
        main = soup.body
    
    text = (main or soup).get_text(separator=" ", strip=True) if soup else ""
    text = re.sub(r"\s+", " ", text).strip()

    title = ""
    if soup and soup.title and soup.title.string:
        title = soup.title.string.strip()
    
    company_name = title if title and len(title) < 60 else url.split("//")[-1].split("/")[0]

    lang = "ru"
    if soup and soup.html:
        html_lang = soup.html.get("lang")
        if html_lang:
            lang = html_lang.lower().split("-")[0]

    return {"text": text, "company_name": company_name, "lang": lang}

def smart_truncate(text: str, max_chars: int = 4500) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_end = max(truncated.rfind(". "), truncated.rfind("! "), truncated.rfind("? "))
    if last_end != -1:
        return truncated[:last_end + 1]
    return truncated

UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]

async def fetch_html_best_effort(url: str) -> tuple[str, str]:
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, verify=False) as http_client:
        headers = {
            "User-Agent": UA_LIST[0],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        
        try:
            r = await http_client.get(url, headers=headers)
            if r.status_code < 400 and len(r.text) > 500:
                return r.text, str(r.url)
        except Exception:
            pass

        if ALLOW_JINA_FALLBACK:
            try:
                jina_url = f"https://r.jina.ai/{url}"
                jr = await http_client.get(jina_url, headers=headers, timeout=25.0)
                if jr.status_code < 400 and jr.text.strip():
                    safe_text = jr.text.replace("<", "&lt;").replace(">", "&gt;")
                    return f"<html><body><main>{safe_text}</main></body></html>", url
            except Exception:
                pass

    raise HTTPException(status_code=503, detail="Не удалось загрузить контент сайта")

# --- 9. Эндпоинты ---

@app.get("/")
async def root():
    return {"service": "Silvia AI API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    raw_url = req.url.strip()
    if not raw_url: raise HTTPException(400, "URL пустой")
    
    url = normalize_url(raw_url)
    if not is_valid_url(url): raise HTTPException(400, "Некорректный URL")

    try:
        html, final_url = await fetch_html_best_effort(url)
        data = extract_main_content(html, final_url)
        
        if not data["text"] or len(data["text"]) < 50:
            raise HTTPException(400, "Сайт пуст или защищен")

        document = smart_truncate(data["text"], max_chars=5000)
        
        return AnalyzeResponse(
            url=final_url,
            document=document,
            company_name=data["company_name"],
            lang=data["lang"],
        )
    except HTTPException: raise
    except Exception as e:
        logger.error(f"Analyze error: {e}")
        raise HTTPException(502, f"Ошибка анализа: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    question = req.question.strip()
    document = req.document.strip()
    company_name = req.company_name or "Компании"
    
    # Получаем историю и ограничиваем её (последние 6 сообщений), 
    # чтобы не отправлять слишком много текста и не тратить токены
    history_messages = req.history[-6:] if req.history else []
    
    if not question or not document:
        raise HTTPException(status_code=400, detail="Нет вопроса или контекста")

    system_prompt = f"""
Ты — AI-консультант сайта "{company_name}".
Твоя база знаний — только текст ниже.
Отвечай вежливо, кратко и по делу. Учитывай предыдущий контекст беседы.

База знаний:
{document[:3500]} 
"""

    # Формируем список сообщений для OpenAI
    messages_payload = [{"role": "system", "content": system_prompt}]
    
    # Добавляем историю диалога
    for msg in history_messages:
        # Защита: разрешаем только роли user и assistant
        if msg.role in ["user", "assistant"]:
            messages_payload.append({"role": msg.role, "content": msg.content})
            
    # Добавляем текущий вопрос
    messages_payload.append({"role": "user", "content": question})

    try:
        chat_resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_payload,
            temperature=0.6,
            max_tokens=400,
        )
        
        answer = chat_resp.choices[0].message.content.strip()
        return ChatResponse(answer=answer)

    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        raise HTTPException(status_code=503, detail="Ошибка AI")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
