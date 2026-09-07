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
TARGET_SAMPLES = 20000  # Change to 20000 for the grand publication run
ADAPTER_PATH = "qwen2-vl-argus-200gb-checkpoints/final_adapter"
OUTPUT_CSV = "argus_vs_moondream_diverse_log.csv"

print("=" * 90)
print("     A.R.G.U.S. vs MOONDREAM2: DIVERSE MULTIMODAL BENCHMARK (VQAv2)     ")
print("=" * 90)

def classify_question(q_text):
    q = q_text.lower().strip()
    if q.startswith("what color"): return "Color / Attribute"
    elif q.startswith("how many") or q.startswith("how much") or "number of" in q: return "Counting / Quantity"
    elif q.startswith("where") or "location" in q or "standing on" in q: return "Spatial / Location"
    elif any(q.startswith(w) for w in ["is ", "are ", "do ", "does ", "can ", "has ", "was ", "were "]): return "Binary (Yes/No)"
    elif "doing" in q or "action" in q or "sport" in q or "playing" in q: return "Action / Activity"
    else: return "General Reasoning"

# This lightweight list will hold all our data in RAM without storing heavy images
eval_data = []

# ==========================================
# 2. PHASE 1: A.R.G.U.S. EVALUATION
# ==========================================
print("\n[PHASE 1] Booting A.R.G.U.S. Pipeline (4-bit NF4)...")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
base_model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", quantization_config=bnb_config, device_map="cuda")
model_argus = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
processor = AutoProcessor.from_pretrained(ADAPTER_PATH)
model_argus.eval()

print("[SYSTEM] Streaming VQAv2 Dataset for A.R.G.U.S...")
dataset_stream_1 = load_dataset("lmms-lab-encoder/VQAv2", split="validation", streaming=True)

start_time_argus = time.time()

for index, item in enumerate(dataset_stream_1):
    if index >= TARGET_SAMPLES: break

    try:
        pil_image = item["image"].convert("RGB")
        question = item["question"]
        q_type = classify_question(question)
        
        valid_human_answers = [str(ans["answer"]).lower().strip() for ans in item["answers"]]
        best_truth = max(set(valid_human_answers), key=valid_human_answers.count)

        system_instruction = "Answer the question directly, concisely, and factually using 1 to 3 words."
        messages = [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": f"{system_instruction}\n\nQuestion: {question}"}]}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(messages)
        inputs = processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to("cuda")

        t0 = time.time()
        with torch.no_grad():
            gen_ids = model_argus.generate(**inputs, max_new_tokens=8, do_sample=False)
        latency_ms = (time.time() - t0) * 1000

        gen_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        argus_pred = processor.batch_decode(gen_trimmed, skip_special_tokens=True)[0].replace('"', '').replace('.', '').strip().lower()
        is_correct = 1 if (argus_pred in valid_human_answers or any(argus_pred == ans for ans in valid_human_answers)) else 0

        # Save to RAM
        eval_data.append({
            "ID": index + 1, "Category": q_type, "Question": question, "Truth": best_truth, "Valid_Answers": valid_human_answers,
            "ARGUS_Pred": argus_pred, "ARGUS_Correct": is_correct, "ARGUS_Latency": latency_ms
        })

        status = "✓" if is_correct else "✗"
        print(f"[A.R.G.U.S.] [{index+1:04d}/{TARGET_SAMPLES}] | [{q_type:<19}] | Q: {question[:30]:<30} | Truth: {best_truth[:10]:<10} | Pred: {argus_pred[:10]:<10} | {status} | ({latency_ms:.1f} ms)")

    except Exception as e:
        print(f"[WARNING] Skipping sample {index + 1}: {e}")
    finally:
        del inputs, gen_ids, gen_trimmed, img_in, vid_in, pil_image
        torch.cuda.empty_cache()

argus_fps = TARGET_SAMPLES / (time.time() - start_time_argus)
argus_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

print("\n[SYSTEM] Tearing down A.R.G.U.S. pipeline to free GPU memory...")
del model_argus, base_model, processor
gc.collect()
torch.cuda.empty_cache()

# ==========================================
# 3. PHASE 2: MOONDREAM2 EVALUATION
# ==========================================
print("\n[PHASE 2] Booting Moondream2 Pipeline (16-bit)...")
torch.cuda.reset_peak_memory_stats()

# We use the latest 2025 revision which utilizes your newly installed pyvips library
model_md = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2", revision="2025-01-09", trust_remote_code=True, device_map="cuda", torch_dtype=torch.float16
)
model_md.eval()

print("[SYSTEM] Re-Streaming VQAv2 Dataset for Moondream2...")
dataset_stream_2 = load_dataset("lmms-lab-encoder/VQAv2", split="validation", streaming=True)

start_time_md = time.time()

for index, item in enumerate(dataset_stream_2):
    if index >= len(eval_data): break

    try:
        pil_image = item["image"].convert("RGB")
        target_record = eval_data[index] # Fetch the exact same record from Phase 1
        prompt = target_record["Question"] + " Answer concisely using only a few words."

        t0 = time.time()
        with torch.no_grad():
            md_pred = model_md.query(pil_image, prompt)["answer"].replace('"', '').replace('.', '').strip().lower()
        latency_ms = (time.time() - t0) * 1000

        is_correct = 1 if (md_pred in target_record["Valid_Answers"] or any(md_pred == ans for ans in target_record["Valid_Answers"])) else 0

        # Update RAM record
        target_record["MD_Pred"] = md_pred
        target_record["MD_Correct"] = is_correct
        target_record["MD_Latency"] = latency_ms

        status = "✓" if is_correct else "✗"
        print(f"[MOONDREAM] [{index+1:04d}/{TARGET_SAMPLES}] | [{target_record['Category']:<19}] | Q: {target_record['Question'][:30]:<30} | Truth: {target_record['Truth'][:10]:<10} | Pred: {md_pred[:10]:<10} | {status} | ({latency_ms:.1f} ms)")

    except Exception as e:
        print(f"[WARNING] Skipping Moondream sample {index + 1}: {e}")
    finally:
        del pil_image
        torch.cuda.empty_cache()

md_fps = TARGET_SAMPLES / (time.time() - start_time_md)
md_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

del model_md
gc.collect()
torch.cuda.empty_cache()

# ==========================================
# 4. REPORT GENERATION & DATA EXPORT
# ==========================================
print("\n" + "=" * 90)
print("                 DIVERSE CATEGORY COMPARISON REPORT                 ")
print("=" * 90)

df_all = pd.DataFrame(eval_data)
# Clean up the output CSV by dropping the heavy valid_answers array
df_csv = df_all.drop(columns=["Valid_Answers"])
df_csv.to_csv(OUTPUT_CSV, index=False)

categories = df_all["Category"].unique()
summary_data = []

for cat in categories:
    cat_df = df_all[df_all["Category"] == cat]
    total = len(cat_df)
    
    argus_acc = (cat_df["ARGUS_Correct"].sum() / total) * 100
    md_acc = (cat_df["MD_Correct"].sum() / total) * 100
    
    argus_lat = cat_df["ARGUS_Latency"].mean()
    md_lat = cat_df["MD_Latency"].mean()
    
    summary_data.append({
        "Category": cat, "Count": total, 
        "ARGUS Acc (%)": round(argus_acc, 1), "MD Acc (%)": round(md_acc, 1),
        "ARGUS Lat (ms)": round(argus_lat, 1), "MD Lat (ms)": round(md_lat, 1)
    })

# Add Overall Stats
summary_data.append({
    "Category": "OVERALL BENCHMARK", "Count": len(df_all),
    "ARGUS Acc (%)": round((df_all["ARGUS_Correct"].sum() / len(df_all)) * 100, 1),
    "MD Acc (%)": round((df_all["MD_Correct"].sum() / len(df_all)) * 100, 1),
    "ARGUS Lat (ms)": round(df_all["ARGUS_Latency"].mean(), 1),
    "MD Lat (ms)": round(df_all["MD_Latency"].mean(), 1)
})

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))
print("=" * 90)
df_summary.to_csv("argus_vs_moondream_diverse_summary.csv", index=False)

# ==========================================
# 5. VISUALIZATION EXPORT
# ==========================================
print("\n[SYSTEM] Generating comparative bar charts...")
sns.set_theme(style="whitegrid")

plot_df = df_summary[df_summary["Category"] != "OVERALL BENCHMARK"]
labels = plot_df["Category"].tolist()
argus_scores = plot_df["ARGUS Acc (%)"].tolist()
md_scores = plot_df["MD Acc (%)"].tolist()

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, argus_scores, width, label='A.R.G.U.S. (Ours)', color='#2b5c8f')
rects2 = ax.bar(x + width/2, md_scores, width, label='Moondream2', color='#d95f02')

ax.set_ylabel('Accuracy (%)', fontweight="bold")
ax.set_title('Head-to-Head Multimodal Reasoning Accuracy by Diverse Category', fontweight="bold", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha="right")
ax.legend()

# Attach exact numbers to the top of the bars
for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("argus_vs_moondream_diverse_chart.png", dpi=300)
print("[SYSTEM] Diverse category comparison graph saved as 'argus_vs_moondream_diverse_chart.png'")