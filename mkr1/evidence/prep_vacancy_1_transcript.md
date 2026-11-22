# 🎯 Прогін симуляції інтерв'ю #1: Junior Python Developer (FastAPI / Celery / FinTech)

- **Компанія:** FinTech Startup на Djinni
- **Вакансія (вхід):** `vacancy_1_djinni_python.txt`
- **Дата прогону:** 2026-11-21 22:54:34
- **Модель:** `openai/gpt-oss-120b` (Groq Cloud API)
- **Затримки:** розбір 3.32 с → генерація 5.93 с → оцінка 12.84 с (разом 22.1 с)
- **Токени:** 2472 prompt + 3232 completion = 5704 (≈ $0.00279)

---

## 1. Структуровані дані, витягнуті з тексту вакансії

```json
{
  "role": "Junior Python Developer (FastAPI / Celery / FinTech)",
  "level": "Junior",
  "tech_stack": [
    "Python 3.10+",
    "FastAPI",
    "Celery",
    "PostgreSQL",
    "Redis",
    "RabbitMQ",
    "Docker",
    "Docker Compose",
    "pytest",
    "mock"
  ],
  "domain": "FinTech",
  "key_responsibilities": [
    "Розробка та підтримка ендпойнтів REST API платіжної платформи",
    "Інтеграція з зовнішніми банківськими провайдерами",
    "Оптимізація швидкодії важких запитів до бази даних",
    "Покриття бізнес-логіки автоматичними unit- та integration-тестами"
  ],
  "highlighted_requirements": [
    "Впевнені знання Python 3.10+ (ООП, typing)",
    "Досвід роботи з асинхронним стеком: FastAPI або aiohttp, async/await",
    "Робота з PostgreSQL (SQL, індекси, транзакції, рівні ізоляції)",
    "Знання черг повідомлень та фонових задач: Celery, Redis або RabbitMQ",
    "Базові навички Docker та Docker Compose",
    "Написання тестів pytest, mock",
    "Розуміння принципів RESTful API, JWT, CORS, Rate Limiting",
    "Досвід роботи з платіжними шлюзами або банківськими API (плюс)",
    "Розуміння мікросервісної архітектури (плюс)",
    "Досвід профілювання продуктивності коду (cProfile, memory_profiler) (плюс)"
  ]
}
```

## 2. 5 згенерованих запитання

### Питання #1 [Async & FastAPI]

**❓** У вас є ендпойнт FastAPI, який викликає зовнішній банківський API через aiohttp і потім записує результат у PostgreSQL у транзакції. Опишіть, як ви організуєте код, щоб уникнути блокування event loop під час очікування відповіді та забезпечити атомарність запису, навіть якщо зовнішній запит зависне.

**🎯 Що перевіряє лід:** Перевірити розуміння неблокуючих I/O, правильного використання async context managers, а також вміння комбінувати async‑await з синхронними транзакціями SQLAlchemy (або asyncpg) і таймаутами.

**🔑 Ключові тези:**
- Використання aiohttp.ClientSession з timeout
- async with session.get(...)
- async with async_session.begin() для транзакції
- Використання asyncio.wait_for або async_timeout
- Обробка asyncio.TimeoutError та rollback транзакції

### Питання #2 [Celery & Queues]

**❓** Потрібно створити задачу Celery, яка обробляє великий CSV‑файл з транзакціями та записує їх у PostgreSQL. Як ви захистите воркер від переповнення пам'яті, якщо файл надходить у 500 МБ, і як будете обробляти випадок, коли RabbitMQ недоступний під час надсилання задачі?

**🎯 Що перевіряє лід:** Оцінити знання про chunk‑processing, streaming файлів, обмеження пам'яті в воркерах, а також стратегії retry/acknowledge і fallback при недоступності брокера.

**🔑 Ключові тези:**
- Читати файл по рядках (iterators) або використовувати pandas.read_csv(chunksize)
- Обмежити розмір повідомлення в Celery (task serializer)
- Використовувати rate_limit або prefetch_limit
- Налаштувати task_acks_late і retry_backoff
- Використовувати Celery's retry() у випадку ConnectionError

### Питання #3 [Database & Transactions]

**❓** У вас є два запити: один читає баланс користувача, інший оновлює його після успішної транзакції. Які рівні ізоляції PostgreSQL ви оберете, щоб уникнути «lost update» і «dirty read», і як реалізуєте це у коді, враховуючи можливі deadlock'и?

**🎯 Що перевіряє лід:** Перевірити знання рівнів ізоляції, їх впливу на фінтех‑операції, а також вміння застосовувати SELECT … FOR UPDATE і обробляти deadlock.

**🔑 Ключові тези:**
- Використовувати рівень REPEATABLE READ або SERIALIZABLE
- SELECT ... FOR UPDATE при читанні балансу
- Обгорнути операції у transaction.atomic (або async_session.begin())
- Обробка psycopg2.errors.DeadlockDetected і повтор запиту

### Питання #4 [Docker & DevOps]

**❓** Опишіть процес створення мінімального Docker‑образу для FastAPI‑сервісу з Celery‑воркером, який використовує лише один базовий образ, і як ви запобігатимете витоку ресурсів (наприклад, зайвих файлів журналу) під час CI/CD.

**🎯 Що перевіряє лід:** Оцінити вміння оптимізувати багатоступеневі образи, розуміння multi‑stage builds, правильного управління томами та логами в контейнерах.

**🔑 Ключові тези:**
- Використання python:3.10‑slim як бази
- Multi‑stage: builder (pip install) → runtime
- COPY --chown=nonroot
- Встановлення gunicorn/uvicorn в entrypoint
- Відключення buffering у логах, перенаправлення в stdout
- Використання docker-compose healthcheck для Celery

### Питання #5 [Testing & Mocking]

**❓** Напишіть схематичний pytest‑тест, який перевіряє, що ваш FastAPI‑ендпойнт правильно обробляє помилку 502 від зовнішнього банківського сервісу, використовуючи mock для aiohttp‑клієнта та перевіряючи, що у відповідь повертається 504 з коректним повідомленням.

**🎯 Що перевіряє лід:** Перевірити практичні навички написання async‑тестів, використання pytest‑asyncio, mock‑об’єктів та правильного маппінгу помилок між зовнішніми та внутрішніми API.

**🔑 Ключові тези:**
- @pytest.mark.asyncio
- async def test_endpoint_error(monkeypatch):
- mock_resp = AsyncMock(status=502)
- monkeypatch.setattr('myapp.http_client.get', mock_resp)
- client = AsyncClient(app=app, base_url='http://test')
- resp = await client.get('/pay')
- assert resp.status_code == 504
- assert resp.json()['detail'] == 'Bank service unavailable'

## 3. Оцінка тестової відповіді (перевірка оцінювача)

**Відповідь кандидата** (складена з ключових тез, які система запропонувала для цього питання — типовий сценарій самопідготовки: подивитись підказки → відповісти своїми словами):

> На співбесіді я б відповів так: Використання aiohttp.ClientSession з timeout; async with session.get(...); async with async_session.begin() для транзакції; Використання asyncio.wait_for або async_timeout; Обробка asyncio.TimeoutError та rollback транзакції. Окремо проговорюю, як це перевірити в проді (метрики, логи, алерти) і які крайові випадки можуть зламати рішення.

- **Оцінка:** `7 / 10`
- **Вердикт:** `Рекомендовано`
- **Red flags (що саме не сподобалось):**
  - Не уточнив, що транзакція має відкриватися **після** успішного зовнішнього запиту, щоб не блокувати DB під час очікування
  - Не розповів про правильне використання async‑SQLAlchemy (engine.begin(), async_session) або asyncpg
  - Не описав, як передати таймаут у саму корутину та як забезпечити коректне скасування (asyncio.shield, cancellation propagation)
  - Не згнав про можливі блокуючі синхронні виклики та їх уникнення

**💡 Еталонна відповідь:** Створюю aiohttp.ClientSession з явно заданим timeout і викликаю зовнішній API через `async with session.get(...)` всередині `asyncio.wait_for`. Після успішного отримання відповіді відкриваю транзакцію через `async with async_engine.begin() as conn` і записую дані, комітуючи лише коли запит завершився без помилок; у випадку `TimeoutError` роблю rollback. Таким чином event loop не блокується, а атомарність забезпечується лише після успішного зовнішнього виклику, а таймаут і скасування коректно обробляються.

---
*Згенеровано TechPrep AI (openai/gpt-oss-120b) о 2026-11-21 22:54:34; цифри не редагувались вручну.*