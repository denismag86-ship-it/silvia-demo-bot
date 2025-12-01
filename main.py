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
from openai import AsyncOpenAI  # Используем тот же SDK!

# --- 1. Логирование ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("silvia")

# --- 2. Конфигурация ---
# 👇 Изменено: теперь читаем DEEPSEEK_API_KEY
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    logger.warning("⚠️ Переменная DEEPSEEK_API_KEY не найдена! Чат работать не будет.")

# 👇 Базовый URL DeepSeek API
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 👇 Модель: deepseek-chat (V3) или deepseek-reasoner (R1)
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

ALLOW_JINA_FALLBACK = os.getenv("ALLOW_JINA_FALLBACK", "1") == "1"

# --- 3. Инициализация FastAPI ---
app = FastAPI(title="Silvia API", version="2.3.0")  # Обновил версию

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

# --- 5. Клиент DeepSeek (через OpenAI SDK) ---
# 👇 Ключевое изменение: указываем base_url и api_key для DeepSeek
client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

# --- 6. Модели данных (Pydantic) --- (без изменений)

class AnalyzeRequest(BaseModel):
    url: str

class AnalyzeResponse(BaseModel):
    url: str
    document: str
    company_name: str
    lang: str

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    document: str
    history: List[Message] = []
    company_name: Optional[str] = None
    lang: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str

# --- 7. Middleware для логирования --- (без изменений)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    path = request.url.path
    method = request.method
    
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

# --- 8. Утилиты --- (без изменений, пропускаю для краткости)
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
    return {
        "service": "Silvia AI API",
        "status": "running",
        "model": DEEPSEEK_MODEL,  # 👈 Добавил для отладки
    }

@app.get("/health")
async def health():
    return {"status": "ok", "provider": "deepseek"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    raw_url = req.url.strip()
    if not raw_url:
        raise HTTPException(400, "URL пустой")
    
    url = normalize_url(raw_url)
    if not is_valid_url(url):
        raise HTTPException(400, "Некорректный URL")

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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analyze error: {e}")
        raise HTTPException(502, f"Ошибка анализа: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    question = req.question.strip()
    document = req.document.strip()
    company_name = req.company_name or "Компании"
    
    history_messages = req.history[-6:] if req.history else []
    
    if not question or not document:
        raise HTTPException(status_code=400, detail="Нет вопроса или контекста")

    system_prompt = f"""
system_prompt = f"""
You are an AI assistant representing "{company_name}" — friendly, professional, and genuinely helpful.

## Your Role
You're here to demonstrate what a smart AI employee can do. The person testing you is exploring AI automation for their business, so show them how natural and effective this can be.

## Your Knowledge Base
Everything you know comes from this text:
{document[:3500]}

## How to Respond (Output in Russian)
- Be conversational and warm, like a knowledgeable colleague, not a robot
- Keep answers clear and concise (2-4 sentences max)
- If you don't know something from the knowledge base, say it honestly: "Я не нашел эту информацию на сайте, но могу помочь с тем, что есть"
- Reference the conversation history naturally — remember what was discussed
- Ask clarifying questions when needed: "Уточните, пожалуйста, вас интересует...?"
- Vary your sentence length and structure
- Use appropriate emojis sparingly (1-2 max) for friendliness

## What NOT to Do
- Don't make up information not in your knowledge base
- Don't use corporate jargon or overly formal language
- Don't write long walls of text
- Don't ignore the conversation context

Think of yourself as a helpful team member who genuinely cares about giving useful answers.
"""

База знаний:
{document[:3500]}
"""

    messages_payload = [{"role": "system", "content": system_prompt}]
    
    for msg in history_messages:
        if msg.role in ["user", "assistant"]:
            messages_payload.append({"role": msg.role, "content": msg.content})
            
    messages_payload.append({"role": "user", "content": question})

    try:
        # 👇 Единственное изменение — модель
        chat_resp = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,  # deepseek-chat или deepseek-reasoner
            messages=messages_payload,
            temperature=0.6,
            max_tokens=400,
        )
        
        answer = chat_resp.choices[0].message.content.strip()
        return ChatResponse(answer=answer)

    except Exception as e:
        logger.error(f"DeepSeek error: {e}")
        raise HTTPException(status_code=503, detail="Ошибка AI")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

