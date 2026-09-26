import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

# Завантажуємо метадані
with open("model_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

feature_cols = meta["encoded_feature_cols"]
scaler_mean = np.array(meta["scaler_mean"])
scaler_scale = np.array(meta["scaler_scale"])
SCALE = meta["price_scale"]
n_features = len(feature_cols)

device = torch.device("cpu")

# Завантажуємо натреновані моделі
def load_all_models():
    # Linear
    lin = nn.Linear(n_features, 1)
    lin.load_state_dict(torch.load("models/linear.pt", map_location=device, weights_only=True))
    lin.eval()

    # nn_v1
    v1 = nn.Sequential(
        nn.Linear(n_features, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    v1.load_state_dict(torch.load("models/nn_v1.pt", map_location=device, weights_only=True))
    v1.eval()

    # nn_v2, v3, v4, v5
    def make_v2():
        return nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )

    v2 = make_v2()
    v2.load_state_dict(torch.load("models/nn_v2.pt", map_location=device, weights_only=True))
    v2.eval()

    return {
        "linear": lin,
        "nn_v1 (best)": v1,
        "nn_v2": v2
    }

models = load_all_models()

def predict_all(*vals):
    # стандартизуємо вхідні дані
    x_raw = np.array(vals, dtype=np.float32)
    x_scaled = (x_raw - scaler_mean) / scaler_scale
    x_t = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)

    preds = {}
    for name, m in models.items():
        with torch.no_grad():
            preds[name] = max(0.0, m(x_t).item() * SCALE)
    return preds

def run_app(selected_model, *vals):
    all_preds = predict_all(*vals)
    sel_pred = all_preds[selected_model]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(all_preds.keys(), all_preds.values(), color=['#7f7f7f', '#2ca02c', '#1f77b4'])
    ax.set_ylabel("Ціна")
    ax.set_title("Порівняння прогнозів моделей")
    plt.tight_layout()

    return f"{sel_pred:,.2f}", fig

# Слайдери для ознак, як у демо лекції
sliders = [
    gr.Slider(
        minimum=-2.0,
        maximum=50.0 if "GB" in col else (5.0 if "kg" in col or "Freq" in col else 1.0),
        value=8.0 if "RAM" in col else (256.0 if "SSD" in col else 0.0),
        label=col
    )
    for col in feature_cols
]

with gr.Blocks(title="Laptop Price - Gradio") as demo:
    gr.Markdown("## Прогноз ціни ноутбука - порівняння моделей")
    with gr.Row():
        with gr.Column():
            model_choice = gr.Dropdown(choices=list(models.keys()), value="nn_v1 (best)", label="Модель")
            submit_btn = gr.Button("Порахувати прогноз", variant="primary")
            for s in sliders:
                s.render()
        with gr.Column():
            pred_out = gr.Textbox(label="Прогноз обраної моделі")
            plot_out = gr.Plot(label="Графік порівняння")

    submit_btn.click(fn=run_app, inputs=[model_choice] + sliders, outputs=[pred_out, plot_out])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
