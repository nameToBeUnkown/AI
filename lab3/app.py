import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from tokenizers import Tokenizer
from transformers import pipeline

device = torch.device("cpu")

# --- 1. Модель Mini-GPT ---
class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_size, context_len, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer('causal_mask', torch.tril(torch.ones(context_len, context_len)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        weights = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.causal_mask[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        return weights @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, context_len, dropout=0.1):
        super().__init__()
        head_size = embed_dim // n_heads
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_size, context_len, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, context_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.sa = MultiHeadAttention(embed_dim, n_heads, context_len, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffwd = FeedForward(embed_dim, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, n_layers, context_len, dropout=0.1):
        super().__init__()
        self.context_len = context_len
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(context_len, embed_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, n_heads, context_len, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

# Завантаження токенізатора і моделей
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
tokenizer_path = os.path.join(MODELS_DIR, "gpt_tokenizer.json")

if os.path.exists(tokenizer_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
else:
    tokenizer = None
    vocab_size = 1200

# Завантажуємо minigpt_v1 і minigpt_v2
model_v1 = MiniGPT(vocab_size=vocab_size, embed_dim=64, n_heads=4, n_layers=2, context_len=32)
model_v2 = MiniGPT(vocab_size=vocab_size, embed_dim=96, n_heads=4, n_layers=3, context_len=48)

p_v1 = os.path.join(MODELS_DIR, "minigpt_v1.pt")
p_v2 = os.path.join(MODELS_DIR, "minigpt_v2.pt")
if os.path.exists(p_v1):
    model_v1.load_state_dict(torch.load(p_v1, map_location=device, weights_only=True))
    model_v1.eval()
if os.path.exists(p_v2):
    model_v2.load_state_dict(torch.load(p_v2, map_location=device, weights_only=True))
    model_v2.eval()

# Завантажуємо модель сентименту
sentiment_pipe = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis")

def analyze_sentiment(text):
    if not text.strip():
        return "Введіть текст для аналізу", 0.0, {}
    res = sentiment_pipe(text)[0]
    label_map = {
        "Very Negative": "Дуже негативний 😡",
        "Negative": "Негативний 🙁",
        "Neutral": "Нейтральний 😐",
        "Positive": "Позитивний 🙂",
        "Very Positive": "Дуже позитивний 😃"
    }
    ukr_label = label_map.get(res['label'], res['label'])
    score = round(float(res['score']), 4)
    summary = f"Вердикт: {ukr_label} (Впевненість: {score * 100:.1f}%)"
    return summary, {ukr_label: score}

def generate_continuation(prompt, model_choice, max_tokens, temperature, top_p):
    if not tokenizer:
        return "Помилка: файл токенізатора не знайдено."
    if not prompt.strip():
        prompt = "Україна"

    model = model_v2 if "v2" in model_choice else model_v1
    model.eval()

    enc = tokenizer.encode(prompt).ids
    if not enc:
        enc = [1]
    idx = torch.tensor([enc], dtype=torch.long)

    with torch.no_grad():
        for _ in range(int(max_tokens)):
            idx_cond = idx[:, -model.context_len:]
            logits = model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_tok), dim=1)

    generated = tokenizer.decode(idx[0].tolist())
    return generated

# --- Створення Gradio UI ---
with gr.Blocks(title="Лабораторна 3: Український NLP помічник") as demo:
    gr.Markdown("# 🇺🇦 Український NLP Асистент (Лабораторна робота 3)")
    gr.Markdown("Інтерактивний демонстраційний застосунок: аналіз емоційного забарвлення тексту та генерація продовження натренованими трансформерами.")

    with gr.Tab("Аналіз тональності (Sentiment Analysis)"):
        gr.Markdown("### Перевірка настрою новини, коментаря або відгуку")
        with gr.Row():
            with gr.Column():
                sentiment_input = gr.Textbox(
                    label="Введіть текст українською",
                    placeholder="Наприклад: Збройні сили України успішно звільняють наші міста та села!",
                    lines=4
                )
                analyze_btn = gr.Button("Аналізувати тональність", variant="primary")
            with gr.Column():
                sentiment_verdict = gr.Textbox(label="Результат класифікації", interactive=False)
                sentiment_label_dist = gr.Label(label="Ймовірність категорій")
        analyze_btn.click(analyze_sentiment, inputs=[sentiment_input], outputs=[sentiment_verdict, sentiment_label_dist])

    with gr.Tab("Генерація тексту (mini-GPT)"):
        gr.Markdown("### Авторегресивне продовження тексту натренованою моделлю")
        with gr.Row():
            with gr.Column():
                gen_prompt = gr.Textbox(
                    label="Початок тексту (промпт)",
                    placeholder="Україна або Новини сьогодні:",
                    lines=2
                )
                model_sel = gr.Radio(
                    ["minigpt_v1 (2 шари, ctx=32)", "minigpt_v2 (3 шари, ctx=48) [Найкраща]"],
                    label="Оберіть чекпоінт моделі",
                    value="minigpt_v2 (3 шари, ctx=48) [Найкраща]"
                )
                tokens_slider = gr.Slider(minimum=10, maximum=80, value=35, step=5, label="Кількість нових токенів")
                temp_slider = gr.Slider(minimum=0.2, maximum=1.4, value=0.7, step=0.1, label="Температура (Sampling Temperature)")
                topp_slider = gr.Slider(minimum=0.5, maximum=1.0, value=0.9, step=0.05, label="Top-P (Nucleus Sampling)")
                gen_btn = gr.Button("Згенерувати продовження", variant="primary")
            with gr.Column():
                gen_output = gr.Textbox(label="Згенерований текст", lines=8, interactive=False)
        gen_btn.click(generate_continuation, inputs=[gen_prompt, model_sel, tokens_slider, temp_slider, topp_slider], outputs=[gen_output])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
