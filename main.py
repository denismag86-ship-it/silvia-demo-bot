import os
import re
import time
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
    """Извлечение основного контента из HTML"""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    # Удаляем шум
    for tag in soup(["script", "style", "nav", "footer", "aside", "header", "form", "button", "img", "svg", "noscript"]):
        tag.decompose()

    # Ищем основной контент
    main = soup.find("main") or soup.find("article") or soup.find("section") or soup.find("div", class_=re.compile(r"content|main", re.I))
    if not main and soup.body:
        main = soup.body
    
    text = (main or soup).get_text(separator=" ", strip=True) if soup else ""
    text = re.sub(r"\s+", " ", text).strip()

    # Извлекаем название компании
    title = ""
    if soup and soup.title and soup.title.string:
        title = soup.title.string.strip()
    
    company_name = title if title else url.split("//")[-1].split("/")[0]

    # Определяем язык
    lang = "ru"
    if soup and soup.html:
        html_lang = soup.html.get("lang")
        if html_lang:
            lang = html_lang.lower().split("-")[0]

    return {"text": text, "company_name": company_name, "lang": lang}

def smart_truncate(text: str, max_chars: int = 2800) -> str:
    """Умное обрезание текста по границам предложений"""
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
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

async def fetch_html_best_effort(url: str) -> tuple[str, str]:
    """
    Возвращает (html, final_url). Несколько попыток с разными UA.
    """
    async with httpx.AsyncClient(
        timeout=30.0, 
        follow_redirects=True,
        verify=False  # Для сайтов с проблемными SSL
    ) as http_client:
        
        # Попытка 1: Несколько User-Agent
        for ua in UA_LIST:
            headers = {
                "User-Agent": ua,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
                "Accept-Encoding": "gzip, deflate, br",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Connection": "keep-alive",
            }
            try:
                logger.info(f"Попытка загрузки с UA: {ua[:50]}...")
                r = await http_client.get(url, headers=headers)
                
                if r.status_code == 200 and r.text.strip():
                    logger.info(f"✅ Успешная загрузка: {url}")
                    return r.text, str(r.url)
                
                if r.status_code < 400 and r.text.strip():
                    logger.info(f"✅ Загрузка с кодом {r.status_code}: {url}")
                    return r.text, str(r.url)
                
                logger.warning(f"❌ Статус {r.status_code} для {url}")
                
            except httpx.TimeoutException:
                logger.warning(f"⏱️ Таймаут для {url} с UA {ua[:30]}...")
                continue
            except Exception as e:
                logger.warning(f"❌ Ошибка для {url}: {str(e)[:100]}")
                continue

        # Попытка 2: HTTP fallback (если был HTTPS)
        if url.startswith("https://"):
            alt_url = url.replace("https://", "http://", 1)
            logger.info(f"Попытка HTTP fallback: {alt_url}")
            
            try:
                headers = {
                    "User-Agent": UA_LIST[0],
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                }
                r = await http_client.get(alt_url, headers=headers)
                if r.status_code < 400 and r.text.strip():
                    logger.info(f"✅ HTTP fallback успешен: {alt_url}")
                    return r.text, alt_url
            except Exception as e:
                logger.warning(f"❌ HTTP fallback failed: {str(e)[:100]}")

        # Попытка 3: Jina AI Reader (если включен)
        if ALLOW_JINA_FALLBACK:
            try:
                logger.info(f"Попытка Jina AI fallback для {url}")
                jina_url = f"https://r.jina.ai/{url}"
                
                headers = {"User-Agent": UA_LIST[0]}
                jr = await http_client.get(jina_url, headers=headers, timeout=30.0)
                
                if jr.status_code < 400 and jr.text.strip():
                    logger.info(f"✅ Jina AI fallback успешен")
                    # Оборачиваем текст в HTML
                    safe_text = jr.text.replace("<", "&lt;").replace(">", "&gt;")
                    html = f"<html><head><title>Content</title></head><body><main>{safe_text}</main></body></html>"
                    return html, url
                    
            except Exception as e:
                logger.warning(f"❌ Jina fallback failed: {str(e)[:100]}")

    # Все попытки провалились
    logger.error(f"❌ Не удалось загрузить {url} ни одним способом")
    raise HTTPException(
        status_code=403, 
        detail="Не удалось загрузить сайт. Возможно, он блокирует автоматические запросы."
    )

# --- Эндпоинты ---
@app.get("/")
@app.head("/")
async def root():
    return {
        "status": "ok", 
        "service": "Silvia API (stateless)", 
        "version": "2.0.0", 
        "endpoints": ["/analyze", "/chat", "/health"]
    }

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
    
    if not raw_url:
        raise HTTPException(status_code=400, detail="URL не может быть пустым")
    
    url = normalize_url(raw_url)
    logger.info(f"📊 Анализ URL: {url}")

    if not is_valid_url(url):
        raise HTTPException(status_code=400, detail="Некорректный URL")

    try:
        html, final_url = await fetch_html_best_effort(url)
        
        if not html or len(html) < 100:
            raise HTTPException(status_code=400, detail="Получен пустой или слишком короткий контент")
        
        data = extract_main_content(html, final_url)
        
        if not data["text"] or len(data["text"]) < 50:
            raise HTTPException(status_code=400, detail="Не удалось извлечь осмысленный контент с сайта")

        document = smart_truncate(data["text"], max_chars=2800)
        
        logger.info(f"✅ Анализ завершен: {len(document)} символов")
        
        return AnalyzeResponse(
            url=final_url,
            document=document,
            company_name=data["company_name"],
            lang=data["lang"],
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка анализа: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Не удалось обработать сайт: {str(e)[:200]}")

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    question = (req.question or "").strip()
    document = (req.document or "").strip()
    company_name = (req.company_name or "вашей компании").strip()
    lang = (req.lang or "ru").strip().split("-")[0]

    if not question:
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")
    
    if not document:
        raise HTTPException(
            status_code=400, 
            detail="Документ пуст. Сначала вызовите /analyze для получения контента сайта."
        )

    logger.info(f"💬 Chat запрос: {question[:100]}...")

    # Проверка на приветствие
    q_lower = question.lower()
    greeting_words = ["привет", "здравствуй", "здрав", "hi", "hello", "hey", "добрый день", "доброе утро", "добрый вечер"]
    
    if any(word in q_lower for word in greeting_words):
        if lang == "en":
            welcome = f"Hi! I'm Silvia, the AI assistant for {company_name}. How can I help you today?"
        else:
            welcome = f"Здравствуйте! Я — Сильвия, цифровой помощник компании «{company_name}». Чем могу помочь?"
        return ChatResponse(answer=welcome)

    # Системный промпт
    if lang == "en":
        system_prompt = f"""You are Silvia, an intelligent digital assistant for "{company_name}".
Answer ONLY based on the information from the company's website provided below.

Rules:
1) Tone: friendly and professional.
2) Don't make up facts. If there's no data, say: "I don't have that information, but I can check with the team!"
3) Don't say "The website says...". You ARE the voice of the company.
4) Keep answers brief (1-3 sentences) but helpful.
5) For off-topic questions, politely redirect to company-related topics.

Context (don't quote directly):
{document}
"""
    else:
        system_prompt = f"""Вы — Сильвия, интеллектуальный цифровой помощник компании «{company_name}».
Отвечайте ТОЛЬКО на основе информации с сайта компании, предоставленной ниже.

Правила:
1) Тон: дружелюбный и профессиональный.
2) Не выдумывайте факты. Если данных нет — скажите: «Этой информации нет на сайте, но я могу уточнить у команды!»
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
        
        answer = chat_resp.choices[0].message.content.strip() if chat_resp.choices else ""
        
        if not answer:
            answer = "Извините, не удалось сгенерировать ответ. Попробуйте переформулировать вопрос."
        
        logger.info(f"✅ Ответ сгенерирован: {len(answer)} символов")
        return ChatResponse(answer=answer)

    except Exception as e:
        logger.error(f"❌ Ошибка чата: {e}", exc_info=True)
        raise HTTPException(
            status_code=503, 
            detail="Сервис временно недоступен. Попробуйте через несколько секунд."
        )


# --- Запуск (для локальной разработки) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
