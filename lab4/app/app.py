r"""EduForge AI — Локальний навчальний асистент та аналізатор знань
Лабораторна робота 4: Локальний запуск і оптимізація моделей.

Застосунок працює з локальною моделлю через OpenAI-сумісний API (llama-server або Ollama).
Підтримує повноцінне потокове передавання (streaming), динамічне вимірювання швидкості
генерації (tokens/s) і затримки першого токена (TTFT), генерацію карток для повторення,
технічний аналіз та аудит коду, а також перевірку доступності сервера з детальними помилками.

Інструкція запуску:
1. Встановіть необхідні залежності:
   pip install gradio openai

2. Запустіть OpenAI-сумісний сервер (llama-server або Ollama):
   # Приклад з llama-server (рекомендовано для повної оптимізації GPU):
   # Для Git Bash:
   # ./llama_server/llama-server.exe -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf --port 8080 --host 127.0.0.1 -ngl 99 -c 4096 -fa on
   # Для PowerShell / CMD:
   # .\llama_server\llama-server.exe -m models\qwen2.5-1.5b-instruct-q4_k_m.gguf --port 8080 --host 127.0.0.1 -ngl 99 -c 4096 -fa on

   # Або приклад з Ollama:
   ollama run qwen2.5:1.5b

3. Запустіть веб-застосунок:
   python app.py
"""

import json
import os
import re
import sys
import time

if sys.stdout:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr:
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import gradio as gr
from openai import OpenAI

DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
DEFAULT_API_KEY = "local"
DEFAULT_MODEL = "local"

def get_client(base_url):
    return OpenAI(base_url=base_url.strip(), api_key=DEFAULT_API_KEY, timeout=60.0)

def check_server_status(base_url):
    """Перевірка доступності сервера та отримання списку моделей."""
    try:
        start = time.perf_counter()
        client = get_client(base_url)
        models_resp = client.models.list()
        elapsed = (time.perf_counter() - start) * 1000
        models = [m.id for m in models_resp.data]
        return f"🟢 Сервер доступний ({elapsed:.1f} мс). Моделі: {', '.join(models) if models else 'default'}"
    except Exception as e:
        return f"🔴 Сервер недоступний за адресою {base_url}. Помилка: {str(e)}\nБудь ласка, переконайтеся, що llama-server або Ollama запущено!"

# --- Модуль 1: Генерація карток для повторення (Flashcards & Quiz) ---
FLASHCARD_SYSTEM_PROMPT = """Ти — викладач Computer Science. Твоє завдання — створити глибокі та якісні навчальні картки (Active Recall) за наданим текстом.

Вимоги до кожної картки:
1. 'term': конкретний термін або концепт (наприклад: 'FP16', 'BF16', 'Квантизація Q4_K_M').
2. 'question': розгорнуте запитання на розуміння суті (наприклад: 'У чому полягає відмінність між FP16 та BF16?', 'Яким чином формат BF16 запобігає переповненню градієнтів?', 'Скільки пам\'яті заощаджує квантизація GGUF Q4_K_M?').
3. 'answer': вичерпна, зрозуміла відповідь за наданим матеріалом (1-2 речення).

Повертай ТІЛЬКИ валідний JSON у форматі:
{"cards": [{"term": "...", "question": "...", "answer": "..."}]}"""

def render_flashcards_html(raw_text):
    """Парсить JSON (або резервний текст) та перетворює на стильні інтерактивні картки у світлій темі."""
    cards = []
    clean_text = raw_text.strip()
    
    # 1. Спроба розпарсити JSON
    try:
        json_candidate = clean_text
        if "```" in json_candidate:
            json_candidate = re.sub(r"^```(?:json)?\s*", "", json_candidate, flags=re.IGNORECASE)
            json_candidate = re.sub(r"\s*```$", "", json_candidate)
            json_candidate = json_candidate.strip()
            
        data = json.loads(json_candidate)
        if isinstance(data, dict) and "cards" in data:
            cards = data["cards"]
        elif isinstance(data, list):
            cards = data
    except Exception:
        pass

    # 2. Якщо JSON не розпарсився — резервний регулярний вираз для тексту
    if not cards:
        items = re.split(r'\n\s*(?=\d+[\.\)]|\#\#\s*Картка|\bКартка\s*\d+)', clean_text)
        for item in items:
            if not item.strip():
                continue
            qm = re.search(r'(?:Запитання|Питання|лицьова сторона)[^\n:]*:\s*([^\n]+)', item, re.IGNORECASE)
            tm = re.search(r'(?:Ключовий термін|Термін)[^\n:]*:\s*([^\n]+)', item, re.IGNORECASE)
            am = re.search(r'(?:Відповідь|зворотна сторона)[^\n:]*:\s*(.*?)(?=(?:\s*[\*\-•]?\s*(?:Ключовий термін|Термін))|$)', item, re.IGNORECASE | re.DOTALL)
            
            q = qm.group(1).strip() if qm else ""
            t = tm.group(1).strip() if tm else ""
            a = am.group(1).strip() if am else ""
            a = re.sub(r'[\s\*\-•🔑\:]+$', '', a).strip()
            
            if q or a:
                cards.append({"term": t or "Концепт", "question": q or "Питання для самоперевірки", "answer": a or "Немає відповіді"})

    if not cards:
        return raw_text

    # 3. Генерація красивих інтерактивних HTML-карток у світлій темі
    html = ['\n<div style="display: flex; flex-direction: column; gap: 14px; margin-top: 10px;">']
    for idx, c in enumerate(cards, 1):
        term = c.get("term", "Концепт")
        q = c.get("question", "Питання для перевірки")
        a = c.get("answer", "Відповідь")
        
        card_html = (
            f'<div style="background: #ffffff; border: 1px solid #e2e8f0; '
            f'border-left: 6px solid #4f46e5; border-radius: 12px; padding: 16px 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); margin-bottom: 12px;">'
            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">'
            f'<span style="font-size: 13px; font-weight: 700; color: #4f46e5; letter-spacing: 0.5px;">🗂️ КАРТКА #{idx}</span>'
            f'<span style="background: #eef2ff; color: #4338ca; border: 1px solid #c7d2fe; padding: 3px 12px; border-radius: 999px; font-size: 12px; font-weight: 600;">🔑 {term}</span>'
            f'</div>'
            f'<div style="font-size: 15px; font-weight: 600; color: #0f172a; margin-bottom: 12px; line-height: 1.4;">❓ {q}</div>'
            f'<details style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 10px 14px; cursor: pointer;">'
            f'<summary style="font-weight: 600; color: #0284c7; outline: none; user-select: none;">💡 Показати відповідь (перевернути картку)</summary>'
            f'<div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #e2e8f0; color: #334155; font-size: 14px; line-height: 1.5;">{a}</div>'
            f'</details>'
            f'</div>'
        )
        html.append(card_html)
    html.append('</div>\n')
    return '\n'.join(html)

def generate_flashcards(notes_text, num_cards, temperature, max_tokens, base_url):
    if not notes_text.strip():
        yield "⚠️ Будь ласка, введіть конспект або тему для створення карток.", "0.00 с", "0.0 tok/s"
        return

    client = get_client(base_url)
    user_prompt = f"Створи {num_cards} змістовних навчальних карток (JSON з полями 'term', 'question', 'answer') за таким матеріалом:\n\n{notes_text.strip()}"
    
    start_time = time.perf_counter()
    first_token_time = None
    accumulated_text = ""
    token_count = 0
    
    try:
        stream = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": FLASHCARD_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            response_format={"type": "json_object"},
            stream=True,
            stream_options={"include_usage": True}
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                content = chunk.choices[0].delta.content
                accumulated_text += content
                token_count += 1
                
                ttft_str = f"{(first_token_time - start_time):.2f} с" if first_token_time else "..."
                now = time.perf_counter()
                speed = (token_count / (now - first_token_time)) if (first_token_time and now > first_token_time) else 0.0
                yield f"⏳ **Генерація структурованих карток (JSON-потік)...**\n\n```json\n{accumulated_text}\n```", ttft_str, f"{speed:.1f} tok/s"
            
            if chunk.usage:
                token_count = chunk.usage.completion_tokens

        end_time = time.perf_counter()
        ttft_str = f"{(first_token_time - start_time):.2f} с" if first_token_time else "0.00 с"
        total_time = end_time - (first_token_time or start_time)
        final_speed = (token_count / total_time) if total_time > 0 else 0.0
        
        # Після завершення рендеримо повноцінні стильні картки
        rendered_cards = render_flashcards_html(accumulated_text)
        yield rendered_cards, ttft_str, f"{final_speed:.1f} tok/s"

    except Exception as e:
        err_msg = f"❌ Помилка звернення до сервера: {str(e)}\n\nПеревірте, чи запущено сервер на {base_url}."
        yield err_msg, "Помилка", "0.0 tok/s"

# --- Модуль 2: Аналізатор і рев'юєр коду (Code Reviewer) ---
CODE_REVIEW_SYSTEM_PROMPT = """Ти — провідний інженер-програміст (Lead Software Engineer) та експерт з оптимізації алгоритмів.
Проведи ретельне та конструктивне рев'ю коду, наданого користувачем.
Твоя відповідь має складатись із таких блоків:
1. 🔍 **Короткий огляд:** що робить код і його загальна оцінка.
2. ⚠️ **Знайдені проблеми та баги:** логічні помилки, крайові випадки (edge cases), проблеми безпеки або витоки ресурсів.
3. ⚡ **Аналіз складності:** часова (Time Complexity) та просторова (Space Complexity) складність за шкалою Big-O.
4. ✨ **Покращений / виправлений варіант коду:** чистий, оптимізований код з коментарями.
5. 💡 **Рекомендації та кращі практики:** (PEP8, типізація, обробка винятків).
Мова: українська."""

def review_code(code_snippet, task_context, temperature, max_tokens, base_url):
    if not code_snippet.strip():
        yield "⚠️ Будь ласка, введіть код для аналізу.", "0.00 с", "0.0 tok/s"
        return

    client = get_client(base_url)
    user_prompt = f"Контекст задачі: {task_context.strip() if task_context.strip() else 'Не вказано'}\n\nКод для аналізу:\n```\n{code_snippet.strip()}\n```"
    
    start_time = time.perf_counter()
    first_token_time = None
    accumulated_text = ""
    token_count = 0
    
    try:
        stream = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": CODE_REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            stream=True,
            stream_options={"include_usage": True}
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                content = chunk.choices[0].delta.content
                accumulated_text += content
                token_count += 1
                
                ttft_str = f"{(first_token_time - start_time):.2f} с" if first_token_time else "..."
                now = time.perf_counter()
                speed = (token_count / (now - first_token_time)) if (first_token_time and now > first_token_time) else 0.0
                yield accumulated_text, ttft_str, f"{speed:.1f} tok/s"
            
            if chunk.usage:
                token_count = chunk.usage.completion_tokens

        end_time = time.perf_counter()
        ttft_str = f"{(first_token_time - start_time):.2f} с" if first_token_time else "0.00 с"
        total_time = end_time - (first_token_time or start_time)
        final_speed = (token_count / total_time) if total_time > 0 else 0.0
        yield accumulated_text, ttft_str, f"{final_speed:.1f} tok/s"

    except Exception as e:
        err_msg = f"❌ Помилка звернення до сервера: {str(e)}\n\nПеревірте, чи запущено сервер на {base_url}."
        yield err_msg, "Помилка", "0.0 tok/s"

# --- Модуль 3: Інтерактивний діалоговий тьютор (Streaming Chat) ---
TUTOR_PROMPTS = {
    "Дружній ментор (детально з прикладами)": "Ти — терплячий академічний наставник. Пояснюй складні поняття зрозумілою мовою з практичними прикладами та метафорами. Заохочуй допитливість студента.",
    "Лаконічний довідник (швидко і по суті)": "Ти — швидкий технічний довідник. Відповідай максимально стисло, структуровано, без зайвих вступів. Наводь лише факти, формули та визначення.",
    "Суворий екзаменатор (перевірка знань)": "Ти — викладач університету на усному іспиті. Оцінюй відповіді студента критично, вказуй на найменші неточності та став провокаційні уточнюючі запитання."
}

def chat_stream(message, history, persona, temperature, top_p, max_tokens, base_url):
    if not message.strip():
        yield history, "", "0.00 с", "0.0 tok/s"
        return

    client = get_client(base_url)
    system_text = TUTOR_PROMPTS.get(persona, TUTOR_PROMPTS["Дружній ментор (детально з прикладами)"])
    
    messages = [{"role": "system", "content": system_text}]
    is_dict_format = False
    if history and isinstance(history[0], dict):
        is_dict_format = True
        messages.extend(history)
    elif history:
        for item in history:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                if item[0]:
                    messages.append({"role": "user", "content": str(item[0])})
                if item[1]:
                    messages.append({"role": "assistant", "content": str(item[1])})
    else:
        is_dict_format = True
    
    messages.append({"role": "user", "content": message})

    if is_dict_format:
        history = list(history or []) + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    else:
        history = list(history or []) + [[message, ""]]

    start_time = time.perf_counter()
    first_token_time = None
    accumulated_text = ""
    token_count = 0

    try:
        stream = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
            stream=True,
            stream_options={"include_usage": True}
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                content = chunk.choices[0].delta.content
                accumulated_text += content
                token_count += 1
                if is_dict_format:
                    history[-1]["content"] = accumulated_text
                else:
                    history[-1][1] = accumulated_text
                
                ttft_str = f"{(first_token_time - start_time):.2f} с" if first_token_time else "..."
                now = time.perf_counter()
                speed = (token_count / (now - first_token_time)) if (first_token_time and now > first_token_time) else 0.0
                yield history, "", ttft_str, f"{speed:.1f} tok/s"

            if chunk.usage:
                token_count = chunk.usage.completion_tokens

        end_time = time.perf_counter()
        ttft_str = f"{(first_token_time - start_time):.2f} с" if first_token_time else "0.00 с"
        total_time = end_time - (first_token_time or start_time)
        final_speed = (token_count / total_time) if total_time > 0 else 0.0
        yield history, "", ttft_str, f"{final_speed:.1f} tok/s"

    except Exception as e:
        err_msg = f"❌ Помилка генерації: {str(e)}\n\nПеревірте сервер на {base_url}."
        if is_dict_format:
            history[-1]["content"] = err_msg
        else:
            history[-1][1] = err_msg
        yield history, "", "Помилка", "0.0 tok/s"

# --- Інтерфейс Gradio Blocks ---
custom_css = """
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; font-family: 'Segoe UI', system-ui, sans-serif; }
.metric-box { border: 1px solid #e0e4eb; border-radius: 8px; padding: 10px; background: #f8fafc; text-align: center; }
"""

with gr.Blocks(title="EduForge AI — Локальний помічник") as demo:
    gr.Markdown("# 🎓 EduForge AI: Локальний навчальний помічник та аналізатор коду")
    gr.Markdown(
        "Інтерактивний демонстраційний застосунок до **Лабораторної роботи №4** (Локальний запуск і оптимізація моделей). "
        "Всі обчислення виконуються **локально** моделлю сімейства Qwen2.5 через OpenAI-сумісний HTTP API."
    )

    with gr.Accordion("⚙️ Налаштування сервера та діагностика підключення", open=False):
        with gr.Row():
            base_url_input = gr.Textbox(value=DEFAULT_BASE_URL, label="API Base URL (разом із /v1)", scale=3)
            check_btn = gr.Button("🔍 Перевірити сервер", scale=1)
        server_status_out = gr.Markdown("Натисніть кнопку вище для перевірки з'єднання з рантаймом.")
        check_btn.click(check_server_status, inputs=[base_url_input], outputs=[server_status_out])

    with gr.Tabs():
        # --- Вкладка 1: Навчальні картки ---
        with gr.TabItem("🗂️ Генератор карток (Active Recall)"):
            gr.Markdown("### Генерація карток для інтервального повторення за конспектом або темою")
            with gr.Row():
                with gr.Column(scale=5):
                    notes_input = gr.Textbox(
                        label="Введіть навчальні нотатки або тему лекції",
                        placeholder="Наприклад: Властивості квантизації моделей. FP16 має 5 біт експоненти і 10 біт мантиси, тоді як BF16 має 8 біт експоненти...",
                        lines=6
                    )
                    with gr.Row():
                        cards_slider = gr.Slider(minimum=3, maximum=8, value=4, step=1, label="Кількість карток")
                        temp_cards = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.05, label="Температура (0 = точність)")
                        tokens_cards = gr.Slider(minimum=300, maximum=1500, value=800, step=50, label="Макс. токенів")
                    gen_cards_btn = gr.Button("⚡ Згенерувати картки для вивчення", variant="primary")
                with gr.Column(scale=6):
                    cards_output = gr.Markdown(label="Згенеровані картки")
                    with gr.Row():
                        cards_ttft = gr.Textbox(label="TTFT (Затримка першого токена)", value="-", interactive=False)
                        cards_speed = gr.Textbox(label="Швидкість генерації", value="-", interactive=False)

            gen_cards_btn.click(
                generate_flashcards,
                inputs=[notes_input, cards_slider, temp_cards, tokens_cards, base_url_input],
                outputs=[cards_output, cards_ttft, cards_speed]
            )

        # --- Вкладка 2: Рев'ю коду ---
        with gr.TabItem("💻 Аудит та оптимізація коду"):
            gr.Markdown("### Пошук багів, оцінка алгоритмічної складності O(N) та рефакторинг коду")
            with gr.Row():
                with gr.Column(scale=5):
                    code_input = gr.Code(
                        label="Вставте вихідний код",
                        language="python",
                        lines=8,
                        value="def find_duplicates(nums):\n    duplicates = []\n    for i in range(len(nums)):\n        for j in range(i + 1, len(nums)):\n            if nums[i] == nums[j] and nums[i] not in duplicates:\n                duplicates.append(nums[i])\n    return duplicates"
                    )
                    task_context_input = gr.Textbox(
                        label="Опис задачі або відома проблема (необов'язково)",
                        placeholder="Знайти всі дублікати в масиві цілих чисел. Працює повільно на великих списках.",
                        lines=2
                    )
                    with gr.Row():
                        temp_code = gr.Slider(minimum=0.0, maximum=0.8, value=0.1, step=0.05, label="Температура (детермінізм)")
                        tokens_code = gr.Slider(minimum=300, maximum=1200, value=650, step=50, label="Макс. токенів")
                    review_btn = gr.Button("🔍 Провести технічний аудит коду", variant="primary")
                with gr.Column(scale=6):
                    review_output = gr.Markdown(label="Результат аудиту")
                    with gr.Row():
                        code_ttft = gr.Textbox(label="TTFT", value="-", interactive=False)
                        code_speed = gr.Textbox(label="Швидкість генерації", value="-", interactive=False)

            review_btn.click(
                review_code,
                inputs=[code_input, task_context_input, temp_code, tokens_code, base_url_input],
                outputs=[review_output, code_ttft, code_speed]
            )

        # --- Вкладка 3: Діалоговий тьютор зі стрімінгом ---
        with gr.TabItem("💬 Діалоговий тьютор (Streaming)"):
            gr.Markdown("### Інтерактивне навчання у режимі реального часу з вимірюванням TTFT")
            with gr.Row():
                with gr.Column(scale=4):
                    persona_sel = gr.Dropdown(
                        choices=list(TUTOR_PROMPTS.keys()),
                        value="Дружній ментор (детально з прикладами)",
                        label="Роль асистента"
                    )
                    temp_chat = gr.Slider(minimum=0.0, maximum=1.2, value=0.3, step=0.05, label="Температура")
                    topp_chat = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-P")
                    tokens_chat = gr.Slider(minimum=100, maximum=1000, value=400, step=50, label="Ліміт токенів")
                    with gr.Row():
                        chat_ttft = gr.Textbox(label="TTFT", value="-", interactive=False)
                        chat_speed = gr.Textbox(label="Швидкість", value="-", interactive=False)
                    clear_btn = gr.Button("🗑️ Очистити діалог")
                with gr.Column(scale=7):
                    chatbot = gr.Chatbot(label="Діалог з моделлю", height=420)
                    chat_msg = gr.Textbox(label="Ваше повідомлення", placeholder="Запитайте щось, наприклад: Чому KV-кеш росте лінійно від контексту?", lines=2)
                    send_btn = gr.Button("Надіслати запит", variant="primary")

            send_btn.click(
                chat_stream,
                inputs=[chat_msg, chatbot, persona_sel, temp_chat, topp_chat, tokens_chat, base_url_input],
                outputs=[chatbot, chat_msg, chat_ttft, chat_speed]
            )
            chat_msg.submit(
                chat_stream,
                inputs=[chat_msg, chatbot, persona_sel, temp_chat, topp_chat, tokens_chat, base_url_input],
                outputs=[chatbot, chat_msg, chat_ttft, chat_speed]
            )
            clear_btn.click(lambda: ([], "-", "-"), outputs=[chatbot, chat_ttft, chat_speed])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft(), css=custom_css)

