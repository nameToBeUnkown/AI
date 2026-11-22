# 🎯 TechPrep AI — Інструкція запуску та використання (Cloud Edition)

**TechPrep AI** — практичний хмарний інструмент підготовки до технічних IT-співбесід на базі штучного інтелекту. Працює на 100% через хмарні API (Groq / OpenAI / Gemini) без завантаження важких моделей та без навантаження локального комп'ютера.

---

## 🛠️ 1. Системні вимоги та встановлення залежностей

Потрібен лише встановлений **Python 3.10+**.

Встановіть легкі залежності проєкту:
```bash
cd mkr1/solution
pip install -r requirements.txt
```

---

## ⚙️ 2. Налаштування Cloud API

Найпростіше — створити `.env` у цій теці (зразок: `.env.example`):
```bash
cp .env.example .env    # і вставити свій ключ
```
`.env` не комітиться в git (`../.gitignore`). Перед здачею роботи файл `.env` з реальним ключем краще видалити — викладач вставить свій ключ у веб-інтерфейсі.

Рішення працює через швидкі хмарні API:

### Рекомендований варіант: Groq API (Швидко і безкоштовно)
1. Отримайте безкоштовний ключ на [console.groq.com](https://console.groq.com/keys) (реєстрація за 30 секунд через Google/GitHub).
2. Задайте змінну оточення або вкажіть її у веб-інтерфейсі:
```bash
# Linux / macOS / Git Bash:
export GROQ_API_KEY="gsk_..."

# Windows PowerShell:
$env:GROQ_API_KEY="gsk_..."
```

### Альтернатива: OpenAI API або Google Gemini API
Також підтримується стандартний `OPENAI_API_KEY`:
```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"
```

Google Gemini підключається через OpenAI-сумісний endpoint (ключ `AIza...`):
```bash
export GEMINI_API_KEY="AIza..."
export OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export OPENAI_MODEL="gemini-3.5-flash"
```
Актуальний список моделей Gemini: `gemini-3.5-flash` (робоча), `gemini-3-flash-preview`, `gemini-2.5-flash`.

---

## 🚀 3. Запуск рішення

### Спосіб 1: Веб-інтерфейс (Gradio Web UI)
Найзручніший спосіб із візуальними картками питань та інтерактивною симуляцією:
```bash
python app.py
```
Відкрийте веб-браузер за адресою:  
👉 **http://127.0.0.1:7861**

Вставте API ключ у верхній панелі, додайте текст будь-якої вакансії та натисніть «Проаналізувати стек та згенерувати питання».

### Спосіб 2: Консольний інтерфейс (CLI)
Для швидкого аналізу з терміналу:
```bash
# Базовий прогін за файлом вакансії зі збереженням шпаргалки
python cli.py -f ../evidence/vacancy_1_djinni_python.txt --count 5 --export-md interview_prep.md

# Інтерактивна репетиція співбесіди прямо в консолі
python cli.py -f ../evidence/vacancy_1_djinni_python.txt --interactive
```
