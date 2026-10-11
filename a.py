import os
import gc
import time
import torch
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig,
    AutoModelForCausalLM
)
from qwen_vl_utils import process_vision_info
from peft import PeftModel

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION & CLASSIFIER
# ==========================================
MODEL_TARGETS = {
    "ARGUS": 50000,
    "Moondream2": 50000,
    "Phi3.5": 20000
}

ARGUS_ADAPTER_PATH = "argus_phased_adapters/part_5_adapter"
DATASET_NAME = "lmms-lab-encoder/VQAv2"
OUTPUT_DIR = "vlm_benchmark_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 160)
print("      A.R.G.U.S. GRAND 3-WAY VLM BENCHMARK EXECUTION ENGINE      ")
print("=" * 160)

def classify_question(q_text: str) -> str:
    q = q_text.lower().strip()
    if q.startswith("what color"): return "Color / Attribute"
    elif q.startswith("how many") or q.startswith("how much") or "number of" in q: return "Counting / Quantity"
    elif q.startswith("where") or "location" in q or "standing on" in q: return "Spatial / Location"
    elif any(q.startswith(w) for w in ["is ", "are ", "do ", "does ", "can ", "has ", "was ", "were "]): return "Binary (Yes/No)"
    elif any(w in q for w in ["doing", "action", "sport", "playing"]): return "Action / Activity"
    else: return "General Reasoning"

def clean_answer(pred_text: str) -> str:
    return str(pred_text).replace('"', '').replace('.', '').replace(',', '').strip().lower()

def compute_correctness(pred: str, valid_answers: list) -> int:
    return 1 if (pred in valid_answers or any(pred == ans for ans in valid_answers)) else 0


# ==========================================
# 2. ISOLATED MODEL INTERFACE WRAPPERS
# ==========================================
class ArgusRunner:
    def __init__(self):
        print("\n[INIT] Booting A.R.G.U.S. (Qwen2-VL 2B NF4 + Adapter)...")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", quantization_config=bnb, device_map="cuda"
        )
        self.model = PeftModel.from_pretrained(base, ARGUS_ADAPTER_PATH).eval()
        self.processor = AutoProcessor.from_pretrained(ARGUS_ADAPTER_PATH)

    def predict(self, pil_image, question: str) -> str:
        prompt = "Answer the question directly and factually using 1 to 3 words."
        messages = [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": f"{prompt}\n\nQuestion: {question}"}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(messages)
        inputs = self.processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        gen_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        return self.processor.batch_decode(gen_trimmed, skip_special_tokens=True)[0]


class MoondreamRunner:
    def __init__(self):
        print("\n[INIT] Booting Moondream2 (16-bit)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2", revision="2025-01-09", trust_remote_code=True,
            device_map="cuda", torch_dtype=torch.float16
        ).eval()

    def predict(self, pil_image, question: str) -> str:
        prompt = f"{question} Answer concisely using 1 to 3 words."
        with torch.no_grad():
            return self.model.query(pil_image, prompt)["answer"]


class Phi35Runner:
    def __init__(self):
        print("\n[INIT] Booting Phi-3.5-Vision-Instruct (4-bit)...")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3.5-vision-instruct", 
            trust_remote_code=True, 
            quantization_config=bnb, 
            device_map="cuda", 
            torch_dtype=torch.bfloat16, 
            _attn_implementation="eager"
        ).eval()
        self.processor = AutoProcessor.from_pretrained("microsoft/Phi-3.5-vision-instruct", trust_remote_code=True)

    def predict(self, pil_image, question: str) -> str:
        prompt = f"<|user|>\n<|image_1|>\nAnswer the question directly and factually using 1 to 3 words.\n\nQuestion: {question}<|end|>\n<|assistant|>\n"
        inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt").to("cuda")
        inputs = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in inputs.items()}
        
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=False)
            
        gen_trimmed = gen_ids[0][inputs['input_ids'].shape[1]:]
        return self.processor.decode(gen_trimmed, skip_special_tokens=True).strip()


# ==========================================
# 3. BENCHMARK EXECUTION ENGINE (PER-MODEL)
# ==========================================
MODEL_REGISTRY = {
    "ARGUS": ArgusRunner,
    "Moondream2": MoondreamRunner,
    "Phi3.5": Phi35Runner
}

def evaluate_model(model_name: str, runner_cls):
    csv_file = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_benchmark.csv")
    completed_ids = set()
    target_samples = MODEL_TARGETS.get(model_name, 50000)

    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        completed_ids = set(df_existing["ID"].tolist())
        print(f"[{model_name}] Resuming from checkpoint: {len(completed_ids):,} / {target_samples:,} already completed.")

    if len(completed_ids) >= target_samples:
        print(f"[{model_name}] Completed all {target_samples:,} samples. Skipping.")
        return

    runner = runner_cls()
    dataset_stream = load_dataset(DATASET_NAME, split="validation", streaming=True)

    batch_records = []

    try:
        for idx, item in enumerate(dataset_stream):
            if idx >= target_samples:
                break
            if idx + 1 in completed_ids:
                continue

            try:
                pil_image = item["image"].convert("RGB")
                question = item["question"]
                valid_answers = [clean_answer(ans["answer"]) for ans in item["answers"]]
                best_truth = max(set(valid_answers), key=valid_answers.count)
                category = classify_question(question)

                t0 = time.time()
                raw_pred = runner.predict(pil_image, question)
                latency_ms = (time.time() - t0) * 1000

                clean_pred = clean_answer(raw_pred)
                is_correct = compute_correctness(clean_pred, valid_answers)

                record = {
                    "ID": idx + 1, "Category": category, "Question": question,
                    "Truth": best_truth, "Pred": clean_pred,
                    "Correct": is_correct, "Latency_ms": round(latency_ms, 2)
                }
                batch_records.append(record)

                status = "✓" if is_correct else "✗"
                print(f"[{model_name}] [{idx+1:05d}/{target_samples}] | [{category:<19}] | Pred: {clean_pred[:12]:<12} | Truth: {best_truth[:12]:<12} | {status} ({latency_ms:.1f} ms)")

                if len(batch_records) >= 100:
                    pd.DataFrame(batch_records).to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
                    batch_records.clear()

            except Exception as e:
                print(f"[WARNING] Skipping sample {idx + 1}: {e}")
                continue

        if batch_records:
            pd.DataFrame(batch_records).to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
            batch_records.clear()

    finally:
        del runner
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{model_name}] Evaluation complete. Hardware memory reclaimed.")


for name, runner_cls in MODEL_REGISTRY.items():
    evaluate_model(name, runner_cls)


# ==========================================
# 4. AGGREGATE STATS & PRECISE REPORT
# ==========================================
print("\n" + "=" * 160)
print(" " * 55 + "DIVERSE CATEGORY COMPARISON REPORT")
print("=" * 160)

model_results = {}
for name in MODEL_REGISTRY.keys():
    csv_path = os.path.join(OUTPUT_DIR, f"{name.lower()}_benchmark.csv")
    if os.path.exists(csv_path):
        target = MODEL_TARGETS.get(name, 50000)
        df_loaded = pd.read_csv(csv_path)
        # Cap data at target to ensure pure parity
        model_results[name] = df_loaded.head(target)

if not model_results:
    print("[ERROR] No results found to aggregate.")
    exit()

first_model_name = list(model_results.keys())[0]
categories = model_results[first_model_name]["Category"].unique()

summary_rows = []

for cat in list(categories) + ["OVERALL BENCHMARK"]:
    row = {"Category": cat}

    for name, df in model_results.items():
        cat_df = df if cat == "OVERALL BENCHMARK" else df[df["Category"] == cat]
        model_cat_total = len(cat_df)

        # Calculates accuracy based on each model's own individual count
        acc = (cat_df["Correct"].sum() / model_cat_total) * 100 if model_cat_total > 0 else 0.0
        lat = cat_df["Latency_ms"].mean() if model_cat_total > 0 else 0.0

        row[f"{name} Acc (%)"] = acc
        row[f"{name} Lat (ms)"] = lat

    summary_rows.append(row)

df_final_summary = pd.DataFrame(summary_rows)

formatters = {}
for col in df_final_summary.columns:
    if "Acc (%)" in col or "Lat (ms)" in col:
        formatters[col] = lambda x: f"{x:>12.1f}"
    elif col == "Category":
        formatters[col] = lambda x: f"{x:>22}"

print(df_final_summary.to_string(index=False, formatters=formatters, justify='right'))
print("=" * 160)

SUMMARY_CSV = os.path.join(OUTPUT_DIR, "all_models_diverse_summary.csv")
df_final_summary.to_csv(SUMMARY_CSV, index=False)


# ==========================================
# 5. HIGH-RESOLUTION PUBLICATION CHART
# ==========================================
print("\n[SYSTEM] Exporting 3-way comparative visualization...")
sns.set_theme(style="whitegrid")

plot_df = df_final_summary[df_final_summary["Category"] != "OVERALL BENCHMARK"]
labels = plot_df["Category"].tolist()
num_categories = len(labels)
num_models = len(MODEL_REGISTRY)

x = np.arange(num_categories)
total_cluster_width = 0.8
bar_width = total_cluster_width / num_models
# Adjusted to 3 colors to match the 3 models
colors = ['#2b5c8f', '#ff7f0e', '#2ca02c']

fig, ax = plt.subplots(figsize=(16, 7))
all_bar_containers = []

for idx, (name, color) in enumerate(zip(MODEL_REGISTRY.keys(), colors)):
    col = f"{name} Acc (%)"
    if col in plot_df.columns:
        vals = plot_df[col].tolist()
        offset = (idx - (num_models - 1) / 2) * bar_width
        rects = ax.bar(x + offset, vals, bar_width, label=name if name != "ARGUS" else "A.R.G.U.S. (Ours)", color=color, edgecolor="black", linewidth=0.5)
        all_bar_containers.append(rects)

ax.set_ylabel('Accuracy (%)', fontweight="bold", fontsize=12)
ax.set_title('Comprehensive VQAv2 Benchmark: A.R.G.U.S. vs Leading Edge VLMs', fontweight="bold", fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=11)
ax.set_ylim(0, 105)
ax.legend(loc='upper right', frameon=True)

for container in all_bar_containers:
    for rect in container:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f"{height:.0f}%", xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, rotation=90)

plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "all_vlms_diverse_comparison_chart.png")
plt.savefig(chart_path, dpi=300)
print(f"[SYSTEM] High-resolution chart saved to: {chart_path}")