"""Core business logic for TechPrep AI — Mock Interview & Question Predictor.
Runs 100% in the cloud via OpenAI-compatible APIs (Groq, OpenAI, Google Gemini).
No heavy local models or local GPU setup required.
"""

import json
import os
import re
import time
from openai import OpenAI

def _load_env_file():
    """Lightweight .env parser without external dependencies."""
    for candidate in [".env", os.path.join(os.path.dirname(__file__), ".env")]:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k and k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass

_load_env_file()

# Endpoint and default model follow whichever key is present, so OPENAI_API_KEY alone doesn't get sent to Groq
PROVIDER_DEFAULTS = {
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-3.5-flash"),
}

def _detect_provider():
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    return "groq"

_provider = _detect_provider()
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL") or PROVIDER_DEFAULTS[_provider][0]
DEFAULT_API_KEY = (os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
                   or os.environ.get("GEMINI_API_KEY", ""))
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL") or PROVIDER_DEFAULTS[_provider][1]
GROQ_URL = PROVIDER_DEFAULTS["groq"][0]
OPENAI_URL = PROVIDER_DEFAULTS["openai"][0]
GEMINI_URL = PROVIDER_DEFAULTS["gemini"][0]

GROQ_FALLBACK_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
OPENAI_FALLBACK_MODELS = ["gpt-4o-mini", "gpt-4o"]
GEMINI_FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash"]
MAX_OUTPUT_TOKENS = 4096
GEMINI_MAX_OUTPUT_TOKENS = 8192

SYSTEM_PARSER_PROMPT = """Ти — провідний IT-рекрутер і технічний архітектор.
Твоє завдання — проаналізувати текст опису вакансії та витягнути з нього структуровану технічну інформацію.

Повертай ТІЛЬКИ валідний JSON у такому форматі:
{
  "role": "Назва позиції (наприклад: Junior Python Backend Developer)",
  "level": "Trainee / Junior / Middle",
  "tech_stack": ["Python 3.11", "FastAPI", "PostgreSQL", "Docker", "Redis", "Celery"],
  "domain": "FinTech / E-commerce / SaaS / EdTech / Загальний",
  "key_responsibilities": ["Розробка REST API", "Оптимізація SQL-запитів", "Інтеграція платіжних систем"],
  "highlighted_requirements": ["Розуміння асинхронності (asyncio)", "Написання тестів pytest", "Робота з брокерами повідомлень"]
}
"""

SYSTEM_QUESTION_GEN_PROMPT = """Ти — Senior Tech Lead та інтерв'юер з 10-річним досвідом.
Твоє завдання — згенерувати {count} реалістичних, практичних та підступних технічних запитань для технічної співбесіди кандидата за наданим описом вакансії та стеку.

Критичні вимоги до запитань:
1. НІЯКИХ банальних підручникових визначень типу «Що таке цикл for?» або «Що таке ООП?».
2. Запитання повинні бути ПРИКЛАДНИМИ і стосуватися стику технологій із вакансії (наприклад: FastAPI + Celery, PostgreSQL + індекси/транзакції, Docker + оптимізація образів, асинхронний код + блокуючі виклики).
3. Додай запитання на обробку відмов (failure handling) та витік ресурсів.

Повертай ТІЛЬКИ валідний JSON у форматі:
{
  "questions": [
    {
      "id": 1,
      "category": "Architecture & Async / Database / Concurrency / Devops",
      "question": "Текст практичного запитання українською мовою",
      "interviewer_goal": "Чому лід ставить це запитання (що саме він перевіряє: розуміння Event Loop, deadlock у базі тощо)",
      "sample_good_points": ["Ключова теза 1", "Ключова теза 2"]
    }
  ]
}
"""

SYSTEM_EVALUATOR_PROMPT = """Ти — принциповий, але справедливий Senior Tech Lead, який проводить технічне інтерв'ю.
Твоє завдання — оцінити відповідь кандидата на технічне запитання.

Вимоги до оцінки:
1. 'score': число від 1 до 10 (1-3: грубі помилки або «вода», 4-6: знає поверхнево, 7-8: хороша впевнена відповідь, 9-10: зріле інженерне мислення з врахуванням нюансів).
2. 'green_flags': список сильних сторін відповіді (що саме кандидату вдалося добре).
3. 'red_flags': список помилок, неточностей або тривожних сигналів для інтерв'юера (чого не вистачило або що сказано неправильно).
4. 'golden_answer': лаконічна, структурована ідеальна відповідь (до 45 секунд мовлення / 3-4 речення), яку лід очікує почути від сильного розробника.

Повертай ТІЛЬКИ валідний JSON у такому форматі:
{
  "score": 8,
  "verdict": "Рекомендовано / Сумнівно / Відхилено",
  "green_flags": ["Правильно вказав на різницю між...", "Згадав про пул з'єднань"],
  "red_flags": ["Не врахував тайм-аут блокування", "Забув про обробку помилок з'єднання"],
  "golden_answer": "Текст ідеальної лаконічної відповіді."
}
"""

def extract_json(raw_text: str) -> dict:
    """Safely extracts JSON from model output, handling markdown blocks if present."""
    text = raw_text.strip()
    if "```" in text:
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        text = match.group(0)
        
    return json.loads(text)

class TechPrepEngine:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, api_key: str = DEFAULT_API_KEY, model: str = DEFAULT_MODEL):
        self.base_url = (base_url or DEFAULT_BASE_URL).strip()
        self.api_key = (api_key or DEFAULT_API_KEY).strip()
        self.model = (model or DEFAULT_MODEL).strip()
        
        if not self.api_key:
            # Fallback to dummy key if using open proxy or local mock
            self.api_key = "dummy-key"
            
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=40.0)

    def _call_llm(self, system_prompt: str, user_content: str, temperature: float = 0.2) -> tuple[str, dict]:
        # Formulate active fallback list for provider
        models_to_try = [self.model]
        if "groq.com" in self.base_url:
            for fb in GROQ_FALLBACK_MODELS:
                if fb not in models_to_try:
                    models_to_try.append(fb)
        elif "openai.com" in self.base_url:
            for fb in OPENAI_FALLBACK_MODELS:
                if fb not in models_to_try:
                    models_to_try.append(fb)
        elif "generativelanguage" in self.base_url:
            for fb in GEMINI_FALLBACK_MODELS:
                if fb not in models_to_try:
                    models_to_try.append(fb)

        max_out = GEMINI_MAX_OUTPUT_TOKENS if "generativelanguage" in self.base_url else MAX_OUTPUT_TOKENS

        last_err = None
        for current_model in models_to_try:
            try:
                start = time.perf_counter()
                resp = self.client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=temperature,
                    max_tokens=max_out,
                    response_format={"type": "json_object"} if any(k in current_model.lower() for k in ["gpt", "llama", "qwen", "mistral"]) else None
                )
                latency = time.perf_counter() - start
                content = resp.choices[0].message.content or ""
                usage = {
                    "model_used": current_model,
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                    "total_tokens": resp.usage.total_tokens if resp.usage else 0,
                    "latency_sec": round(latency, 2)
                }
                return content, usage
            except Exception as e:
                err_text = str(e).lower()
                # Retired model, per-model token cap or provider overload — try the next one
                if ("does not exist" in err_text or "model_not_found" in err_text
                        or "not found" in err_text or "not supported" in err_text
                        or "do not have access" in err_text or "429" in err_text
                        or "503" in err_text or "high demand" in err_text):
                    last_err = e
                    continue
                raise e

        if last_err:
            raise last_err

    def parse_vacancy(self, vacancy_text: str) -> tuple[dict, dict]:
        if not vacancy_text.strip():
            raise ValueError("Текст вакансії порожній.")
        
        raw_out, usage = self._call_llm(
            SYSTEM_PARSER_PROMPT, 
            f"Проаналізуй цю вакансію:\n\n{vacancy_text.strip()}", 
            temperature=0.1
        )
        data = extract_json(raw_out)
        return data, usage

    def generate_interview_questions(self, vacancy_data: dict, count: int = 5) -> tuple[list, dict]:
        formatted_info = json.dumps(vacancy_data, ensure_ascii=False, indent=2)
        system_prompt = SYSTEM_QUESTION_GEN_PROMPT.replace("{count}", str(count))
        
        user_prompt = (
            f"Ось структуровані вимоги вакансії:\n{formatted_info}\n\n"
            f"Сформуй {count} найбільш релевантних практичних запитань, які ставлять на технічному інтерв'ю."
        )
        raw_out, usage = self._call_llm(system_prompt, user_prompt, temperature=0.3)
        data = extract_json(raw_out)
        questions = data.get("questions", []) if isinstance(data, dict) else data
        return questions, usage

    def evaluate_candidate_answer(self, question: dict, user_answer: str) -> tuple[dict, dict]:
        if not user_answer.strip():
            raise ValueError("Відповідь користувача порожня.")
            
        context_str = f"Питання: {question.get('question')}\nМета інтерв'юера: {question.get('interviewer_goal')}"
        user_prompt = f"Контекст запитання:\n{context_str}\n\nВідповідь кандидата:\n{user_answer.strip()}"
        
        raw_out, usage = self._call_llm(SYSTEM_EVALUATOR_PROMPT, user_prompt, temperature=0.2)
        data = extract_json(raw_out)
        return data, usage

    def export_cheat_sheet_markdown(self, vacancy_data: dict, questions: list) -> str:
        role = vacancy_data.get("role", "Технічна позиція")
        level = vacancy_data.get("level", "Junior")
        domain = vacancy_data.get("domain", "General")
        stack = ", ".join(vacancy_data.get("tech_stack", []))
        
        md = [
            f"# 🎯 Шпаргалка підготовки до співбесіди: {role} ({level})",
            f"**Домен:** {domain} | **Ключовий стек:** `{stack}`\n",
            "---",
            "## 📋 Перелік цільових запитань технічного інтерв'ю\n"
        ]
        
        for q in questions:
            qid = q.get("id", "-")
            cat = q.get("category", "General")
            question_text = q.get("question", "")
            goal = q.get("interviewer_goal", "")
            good_pts = q.get("sample_good_points", [])

            md.append(f"### Питання #{qid} [{cat}]")
            md.append("")
            md.append(f"**❓ Запитання:** {question_text}")
            md.append("")
            md.append(f"**🎯 Що перевіряє лід:** *{goal}*")
            md.append("")
            if good_pts:
                md.append("**🔑 Ключові тези для висвітлення:**")
                for pt in good_pts:
                    md.append(f"- {pt}")
            md.append("")
            
        md.append("---\n*Згенеровано автоматично хмарною системою TechPrep AI.*")
        return "\n".join(md)
