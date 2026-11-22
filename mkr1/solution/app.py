"""Web UI for TechPrep AI using Gradio (100% Cloud-powered).
Gradio 6.0 compatible with persistent browser localStorage for API keys
and smart auto-detection for OpenAI, Groq, and Gemini providers.
"""

import html
import os
import sys

if sys.stdout:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr:
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from core import TechPrepEngine, DEFAULT_BASE_URL, DEFAULT_API_KEY, DEFAULT_MODEL

GROQ_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-120b"

MODEL_CHOICES = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "gpt-4o-mini",
    "gemini-3.5-flash",
]

PROVIDERS = {
    "Groq (Потужний: gpt-oss-120b)": {
        "url": GROQ_URL,
        "model": GROQ_MODEL
    },
    "Groq (Швидкий: gpt-oss-20b)": {
        "url": GROQ_URL,
        "model": "openai/gpt-oss-20b"
    },
    "OpenAI (Рекомендовано: gpt-4o-mini)": {
        "url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini"
    },
    "Google Gemini (gemini-3.5-flash)": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.5-flash"
    },
    "Власний / OpenRouter": {
        "url": "https://openrouter.ai/api/v1",
        "model": GROQ_MODEL
    }
}

current_parsed_info = {}
current_questions = []

def on_provider_change(provider_choice):
    cfg = PROVIDERS.get(provider_choice, PROVIDERS["Groq (Потужний: gpt-oss-120b)"])
    return cfg["url"], gr.update(value=cfg["model"], choices=MODEL_CHOICES)

def on_key_change(key: str, current_prov: str, current_url: str, current_mod: str):
    """Auto-detect provider and model when user enters or restores an API key."""
    key = (key or "").strip()
    if key.startswith("gsk_"):
        return (
            "Groq (Потужний: gpt-oss-120b)",
            GROQ_URL,
            gr.update(value=GROQ_MODEL, choices=MODEL_CHOICES),
            "🟢 **Розпізнано Groq API:** встановлено модель `openai/gpt-oss-120b`."
        )
    elif key.startswith("sk-") and not key.startswith("sk-or-"):
        return (
            "OpenAI (Рекомендовано: gpt-4o-mini)",
            "https://api.openai.com/v1",
            gr.update(value="gpt-4o-mini", choices=MODEL_CHOICES),
            "🟢 **Розпізнано OpenAI API:** модель встановлено на `gpt-4o-mini`."
        )
    elif key.startswith("AIzaSy"):
        return (
            "Google Gemini (gemini-3.5-flash)",
            GEMINI_URL,
            gr.update(value=GEMINI_MODEL, choices=MODEL_CHOICES),
            "🟢 **Розпізнано Google Gemini API:** модель встановлено на `gemini-3.5-flash`."
        )
    return current_prov, current_url, current_mod, "🔒 Ключ активний і зберігається у вашому браузері (localStorage)."

GROQ_MODELS = {GROQ_MODEL, "openai/gpt-oss-20b"}
OPENAI_MODELS = {"gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"}
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_MODELS = {"gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3-flash-preview",
                 "gemini-flash-latest", "gemini-2.5-flash", "gemini-3.8-flash"}

def auto_fix_model_provider(url: str, model: str, api_key: str) -> tuple[str, str, str]:
    """Auto-correct endpoint and model mismatch to prevent 404 model_not_found errors."""
    notice = ""
    url = (url or "").strip()
    model = (model or "").strip()
    api_key = (api_key or "").strip()

    if api_key.startswith("gsk_"):
        if url != GROQ_URL:
            url = GROQ_URL
            notice += f"💡 [Авто-виправлення] Виявлено ключ Groq: встановлено endpoint '{GROQ_URL}'. "
        if model not in GROQ_MODELS:
            model = GROQ_MODEL
            notice += f"💡 Встановлено модель '{GROQ_MODEL}'. "

    elif api_key.startswith("sk-") and not api_key.startswith("sk-or-"):
        if "api.openai.com" not in url:
            url = "https://api.openai.com/v1"
            notice += "💡 [Авто-виправлення] Виявлено ключ OpenAI: встановлено endpoint 'https://api.openai.com/v1'. "
        if model not in OPENAI_MODELS:
            model = "gpt-4o-mini"
            notice += "💡 Встановлено модель 'gpt-4o-mini'. "

    elif api_key.startswith("AIzaSy"):
        url = GEMINI_URL
        if model not in GEMINI_MODELS:
            model = GEMINI_MODEL
            notice += f"💡 [Авто-виправлення] Виявлено ключ Gemini: встановлено Gemini endpoint і модель '{GEMINI_MODEL}'. "

    if not model:
        model = ("gpt-4o-mini" if "api.openai.com" in url else
                 GEMINI_MODEL if "generativelanguage" in url else GROQ_MODEL)
        notice += f"💡 Модель не вказана — використано '{model}'. "

    return url, model, notice.strip()

def render_question_card(question, idx):
    q_text = html.escape(str(question.get("question", "")))
    cat = html.escape(str(question.get("category", "General")))
    goal = html.escape(str(question.get("interviewer_goal", "")))
    return (
        f'<div class="question-card">'
        f'<div class="question-card-title">ПИТАННЯ #{idx} [{cat}]</div>'
        f'<div class="question-card-text">{q_text}</div>'
        f'<div class="question-card-goal">🎯 <b>Чому лід це питає:</b> {goal}</div>'
        f'</div>'
    )

def question_index(selected):
    if selected and "#" in selected:
        try:
            return max(0, int(selected.split(":")[0].replace("#", "")) - 1)
        except ValueError:
            return 0
    return 0

def show_question(selected):
    if not current_questions:
        return ""
    idx = min(question_index(selected), len(current_questions) - 1)
    return render_question_card(current_questions[idx], idx + 1)

def process_vacancy(vacancy_text, provider_choice, base_url, api_key, model_name, q_count):
    global current_parsed_info, current_questions
    if not vacancy_text.strip():
        return "⚠️ Будь ласка, вставте опис вакансії.", "", gr.update(choices=[]), None, "", gr.update()

    if not api_key.strip():
        return "⚠️ Будь ласка, вкажіть ваш API Key у блоці налаштувань вище.", "", gr.update(choices=[]), None, "", gr.update()

    clean_url, clean_model, notice = auto_fix_model_provider(base_url, model_name, api_key)
    client_engine = TechPrepEngine(base_url=clean_url, api_key=api_key, model=clean_model)
    try:
        parsed_data, p_usage = client_engine.parse_vacancy(vacancy_text)
        current_parsed_info = parsed_data
        
        questions, q_usage = client_engine.generate_interview_questions(parsed_data, count=int(q_count))
        current_questions = questions
        
        total_time = p_usage.get('latency_sec', 0) + q_usage.get('latency_sec', 0)
        total_tok = p_usage.get('total_tokens', 0) + q_usage.get('total_tokens', 0)
        notice_line = f"\n> {notice}\n" if notice else ""
        
        role_summary = (
            f"### 💼 Позиція: {parsed_data.get('role')} ({parsed_data.get('level')})\n"
            f"{notice_line}"
            f"**🏢 Домен:** {parsed_data.get('domain')} | **⚡ Модель:** `{clean_model}` | **⏱️ Затримка:** {total_time:.1f} с ({total_tok} токенів)\n\n"
            f"**📦 Стек технологій:** `{', '.join(parsed_data.get('tech_stack', []))}`\n\n"
            f"**🔑 Ключові обов'язки:** {', '.join(parsed_data.get('key_responsibilities', []))}"
        )
        
        q_html = ['<div style="display:flex; flex-direction:column; gap:12px; margin-top:10px;">']
        q_choices = []
        for idx, q in enumerate(questions, 1):
            q_choices.append(f"#{idx}: {q.get('question', '')[:70]}...")
            q_html.append(render_question_card(q, idx))
        q_html.append('</div>')
        
        cheat_sheet_md = client_engine.export_cheat_sheet_markdown(parsed_data, questions)
        first_card = render_question_card(questions[0], 1) if questions else ""
        return role_summary, "\n".join(q_html), gr.update(choices=q_choices, value=q_choices[0] if q_choices else None), cheat_sheet_md, first_card, gr.update(value=clean_model)

    except Exception as e:
        err_str = str(e)
        hint = ""
        if "does not exist" in err_str or "model_not_found" in err_str or "404" in err_str:
            hint = f"\n\n💡 **Підказка:** Такої моделі немає у провайдера. Для Groq доступні `{GROQ_MODEL}` та `openai/gpt-oss-20b`; для OpenAI — `gpt-4o-mini`."
        elif "429" in err_str or "rate_limit" in err_str.lower():
            hint = "\n\n💡 **Підказка:** Вичерпано ліміт токенів/запитів хвилини. Зачекайте ~60 с або зменшіть кількість запитань."
        return f"❌ Помилка обробки хмарним API: {err_str}{hint}", "", gr.update(choices=[]), None, "", gr.update()

def evaluate_answer(selected_q_idx_str, candidate_answer, base_url, api_key, model_name):
    global current_questions
    if not current_questions:
        return "⚠️ Спочатку згенеруйте запитання за вакансією."
    if not candidate_answer.strip():
        return "⚠️ Введіть вашу відповідь на запитання для оцінювання."
    if not api_key.strip():
        return "⚠️ Вкажіть API Key у налаштуваннях."
        
    idx = question_index(selected_q_idx_str)
    if idx >= len(current_questions):
        idx = 0
    target_q = current_questions[idx]
    
    clean_url, clean_model, _ = auto_fix_model_provider(base_url, model_name, api_key)
    client_engine = TechPrepEngine(base_url=clean_url, api_key=api_key, model=clean_model)
    try:
        eval_data, usage = client_engine.evaluate_candidate_answer(target_q, candidate_answer)
        try:
            score = float(eval_data.get("score", 5))
        except (TypeError, ValueError):
            score = 5.0
        verdict = html.escape(str(eval_data.get("verdict", "Не визначено")))
        color = "#16a34a" if score >= 7 else ("#ca8a04" if score >= 5 else "#dc2626")

        gf_items = "".join([f"<li>✅ {html.escape(str(item))}</li>" for item in eval_data.get("green_flags", [])])
        rf_items = "".join([f"<li>⚠️ {html.escape(str(item))}</li>" for item in eval_data.get("red_flags", [])])
        golden = html.escape(str(eval_data.get("golden_answer", "")))

        out_html = f"""
        <div class="eval-card">
          <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(125,125,125,0.25); padding-bottom:12px; margin-bottom:14px;">
            <div style="font-size:18px; font-weight:700;">Вердикт інтерв'юера: <span style="color:{color}; font-weight:800;">{verdict}</span></div>
            <div style="background:{color}; color:#ffffff; font-weight:800; font-size:16px; padding:4px 14px; border-radius:999px;">
              Оцінка: {score:g}/10
            </div>
          </div>

          <div style="margin-bottom:14px;">
            <div style="font-weight:700; color:#22c55e; margin-bottom:6px;">🟢 Сильні сторони (Green Flags):</div>
            <ul style="margin:0; padding-left:20px; font-size:14px; line-height:1.5;">{gf_items}</ul>
          </div>

          <div style="margin-bottom:14px;">
            <div style="font-weight:700; color:#ef4444; margin-bottom:6px;">🔴 Ризики та неточності (Red Flags):</div>
            <ul style="margin:0; padding-left:20px; font-size:14px; line-height:1.5;">{rf_items}</ul>
          </div>

          <div style="background:rgba(34, 197, 94, 0.12); border:1px solid rgba(34, 197, 94, 0.3); border-radius:8px; padding:12px 16px;">
            <div style="font-weight:700; color:#4ade80; margin-bottom:6px;">💡 Як відповісти ідеально за 45 секунд:</div>
            <div style="font-size:14px; line-height:1.5;">{golden}</div>
          </div>
        </div>
        """
        return out_html
    except Exception as e:
        return f"❌ Помилка оцінювання: {str(e)}"

custom_css = """
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }

/* LocalStorage status badge with high contrast in dark mode */
.storage-badge {
    padding: 10px 14px !important;
    border-radius: 8px !important;
    margin: 4px 0 !important;
    border: 1px solid #16a34a !important;
    background-color: #052e16 !important;
    display: block !important;
}
.storage-badge, .storage-badge p, .storage-badge span, .storage-badge strong {
    color: #4ade80 !important;
    font-size: 13.5px !important;
    margin: 0 !important;
    line-height: 1.4 !important;
}
.storage-badge strong {
    color: #86efac !important;
    font-weight: 700 !important;
}

/* Light mode overrides for badge */
:not(.dark) .storage-badge {
    background-color: #f0fdf4 !important;
    border: 1px solid #86efac !important;
}
:not(.dark) .storage-badge, :not(.dark) .storage-badge p, :not(.dark) .storage-badge span, :not(.dark) .storage-badge strong {
    color: #166534 !important;
}

/* Question cards */
.question-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-left: 5px solid #3b82f6;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
}
.question-card-title {
    font-weight: 700;
    color: #60a5fa;
    font-size: 13px;
    margin-bottom: 6px;
}
.question-card-text {
    font-size: 15px;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 8px;
}
.question-card-goal {
    font-size: 13px;
    color: #cbd5e1;
    background: rgba(15, 23, 42, 0.6);
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid #334155;
}

:not(.dark) .question-card {
    background: #ffffff;
    border-color: #e2e8f0;
}
:not(.dark) .question-card-text {
    color: #0f172a;
}
:not(.dark) .question-card-goal {
    color: #334155;
    background: #f8fafc;
    border-color: #e2e8f0;
}

/* Evaluation card */
.eval-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px;
    margin-top: 10px;
    color: #f8fafc;
}
:not(.dark) .eval-card {
    background: #ffffff;
    border-color: #e2e8f0;
    color: #0f172a;
}
"""

# Native HTML/JS injected via launch(head=...) for reliable localStorage sync
head_js = """
<script>
(function() {
    function restoreApiKey() {
        const keyWrapper = document.getElementById("api_key_input");
        if (!keyWrapper) return false;
        const input = keyWrapper.querySelector("input, textarea");
        if (!input) return false;

        try {
            const savedKey = localStorage.getItem("techprep_api_key");
            if (savedKey && !input.value) {
                input.value = savedKey;
                input.dispatchEvent(new Event("input", { bubbles: true }));
                input.dispatchEvent(new Event("change", { bubbles: true }));
            }
            if (!input.dataset.storageAttached) {
                input.dataset.storageAttached = "true";
                input.addEventListener("input", function(e) {
                    if (e.target.value) {
                        localStorage.setItem("techprep_api_key", e.target.value);
                    }
                });
                input.addEventListener("change", function(e) {
                    if (e.target.value) {
                        localStorage.setItem("techprep_api_key", e.target.value);
                    }
                });
            }
        } catch(e) {}
        return true;
    }

    let attempts = 0;
    const interval = setInterval(function() {
        attempts++;
        if (restoreApiKey() || attempts > 30) {
            clearInterval(interval);
        }
    }, 150);
})();
</script>
"""

with gr.Blocks(title="TechPrep AI — Симулятор технічних співбесід (Cloud)") as demo:
    gr.Markdown("# 🎯 TechPrep AI: Хмарний генератор технічних питань та тренажер співбесід")
    gr.Markdown("Модульна контрольна робота №1: «Як я покращив своє життя за допомогою AI». Студент: **Горбачов Олег (Oleh Horbachov)**, група ВТ-23-1.")
    
    with gr.Accordion("⚙️ Налаштування хмарного підключення (Cloud API & LocalStorage)", open=True):
        with gr.Row():
            provider_dropdown = gr.Dropdown(
                label="Оберіть хмарного провайдера",
                choices=list(PROVIDERS.keys()),
                value="Groq (Потужний: gpt-oss-120b)",
                elem_id="provider_dropdown",
                scale=2
            )
            api_key_in = gr.Textbox(
                value=DEFAULT_API_KEY, 
                label="API Key (Groq `gsk_...`, OpenAI `sk-...`, Gemini `AIzaSy...`)", 
                type="password", 
                elem_id="api_key_input",
                placeholder="Вставте ваш ключ — він автоматично зберігається в localStorage вашого браузера",
                scale=2
            )
        with gr.Row():
            base_url_in = gr.Textbox(
                value=GROQ_URL, 
                label="Endpoint URL", 
                elem_id="base_url_input",
                scale=2
            )
            model_in = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=GROQ_MODEL, 
                label="Модель (можна вписати свою)", 
                allow_custom_value=True,
                elem_id="model_input",
                scale=1
            )

        with gr.Row():
            key_status_label = gr.Markdown(
                "🔒 **LocalStorage:** Ваш API-ключ автоматично кешується в браузері. Його більше не потрібно вводити повторно.",
                elem_classes=["storage-badge"]
            )
            clear_key_btn = gr.Button("🗑️ Скинути ключ із браузера", size="sm", scale=1)

        # On provider dropdown change
        provider_dropdown.change(
            on_provider_change,
            inputs=[provider_dropdown],
            outputs=[base_url_in, model_in]
        )

        # On API key input change: auto-detect provider
        api_key_in.change(
            on_key_change,
            inputs=[api_key_in, provider_dropdown, base_url_in, model_in],
            outputs=[provider_dropdown, base_url_in, model_in, key_status_label]
        )

        # Clear key button
        clear_key_btn.click(
            fn=lambda: ("", "Ключ видалено з пам'яті браузера."),
            inputs=None,
            outputs=[api_key_in, key_status_label],
            js="""() => {
                try {
                    localStorage.removeItem('techprep_api_key');
                } catch(e) {}
            }"""
        )

    with gr.Tabs():
        with gr.TabItem("📋 1. Аналіз вакансії та генерація питань"):
            with gr.Row():
                with gr.Column(scale=5):
                    vacancy_input = gr.Textbox(
                        lines=10, 
                        label="Вставте опис вакансії (з Djinni, DOU, LinkedIn або корпоративного сайту)",
                        placeholder="Наприклад:\nJunior Python Developer у компанію...\nВимоги: FastAPI, Docker, PostgreSQL, Celery, розуміння REST та асинхронності..."
                    )
                    q_count_slider = gr.Slider(minimum=3, maximum=8, value=5, step=1, label="Кількість запитань для симуляції")
                    gen_btn = gr.Button("🚀 Проаналізувати стек та згенерувати питання", variant="primary")
                with gr.Column(scale=6):
                    role_summary_out = gr.Markdown(label="Стек вакансії")
                    questions_html_out = gr.HTML(label="Згенеровані запитання")

        with gr.TabItem("🎙️ 2. Інтерактивний тренажер (Mock Interview)"):
            with gr.Row():
                with gr.Column(scale=5):
                    question_selector = gr.Dropdown(label="Оберіть запитання для відповіді", choices=[])
                    question_preview = gr.HTML()
                    candidate_answer_in = gr.Textbox(
                        lines=7, 
                        label="Ваша відповідь на запитання", 
                        placeholder="Поясніть логіку вирішення, архітектурний підхід або наберіть стислу відповідь так, ніби ви на співбесіді..."
                    )
                    eval_btn = gr.Button("🧐 Отримати оцінку Tech Lead", variant="primary")
                with gr.Column(scale=6):
                    evaluation_out = gr.HTML(label="Звіт та рекомендації інтерв'юера")

        with gr.TabItem("💾 3. Експорт персоналізованої шпаргалки"):
            cheat_sheet_viewer = gr.Markdown(label="Повна шпаргалка для повторення перед інтерв'ю")

    gen_btn.click(
        process_vacancy,
        inputs=[vacancy_input, provider_dropdown, base_url_in, api_key_in, model_in, q_count_slider],
        outputs=[role_summary_out, questions_html_out, question_selector, cheat_sheet_viewer, question_preview, model_in]
    )

    question_selector.change(
        show_question,
        inputs=[question_selector],
        outputs=[question_preview]
    )

    eval_btn.click(
        evaluate_answer,
        inputs=[question_selector, candidate_answer_in, base_url_in, api_key_in, model_in],
        outputs=[evaluation_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, theme=gr.themes.Soft(), css=custom_css, head=head_js)
