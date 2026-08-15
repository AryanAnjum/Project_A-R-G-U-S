import os
import torch
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================
TARGET_EVAL_COUNT = 10000 
LOG_CSV = "argus_10k_evaluation_log.csv"
METRICS_CSV = "argus_10k_evaluation_metrics.csv"
HEATMAP_PNG = "argus_10k_confusion_matrix.png"
ADAPTER_PATH = "qwen2-vl-argus-200gb-checkpoints/final_adapter"

print("\n[SYSTEM] Booting Massive Benchmark Evaluator...")

# Initialize per-sample log CSV if not present
if not os.path.exists(LOG_CSV):
    df_empty = pd.DataFrame(columns=["Index", "Question", "Ground_Truth", "AI_Prediction", "Is_Correct", "Question_Type"])
    df_empty.to_csv(LOG_CSV, index=False)
    print(f"[SYSTEM] Initialized evaluation log at: {LOG_CSV}")

# ==========================================
# 2. LOAD QWEN MODEL + LORA ADAPTER
# ==========================================
print("[SYSTEM] Loading Qwen2-VL Base Model (4-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", 
    quantization_config=bnb_config,
    device_map="cuda"
)

print(f"[SYSTEM] Applying Custom Adapter from: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
processor = AutoProcessor.from_pretrained(ADAPTER_PATH)
model.eval()

# ==========================================
# 3. STREAM UNSEEN DATASET (PARQUET STREAM)
# ==========================================
print("[SYSTEM] Connecting to Hugging Face Cloud (VQAv2 Validation Parquet Stream)...")
dataset = load_dataset("lmms-lab-encoder/VQAv2", split="validation", streaming=True)

# Tracking metrics
total_count = 0
correct_count = 0

# Category tracking
cat_stats = {
    "yes/no": {"correct": 0, "total": 0},
    "number": {"correct": 0, "total": 0},
    "other": {"correct": 0, "total": 0}
}

# Binary tracking for Confusion Matrix (Yes/No subset)
yn_true = []
yn_pred = []

print(f"\n[SYSTEM] Starting continuous evaluation for {TARGET_EVAL_COUNT} samples...")
print("[SYSTEM] Progress is auto-saved continuously. Safe to stop anytime with Ctrl+C.\n")

progress_bar = tqdm(total=TARGET_EVAL_COUNT, desc="Evaluating", unit="img")

# ==========================================
# 4. EVALUATION LOOP
# ==========================================
for index, item in enumerate(dataset):
    if index >= TARGET_EVAL_COUNT:
        break

    try:
        pil_image = item["image"].convert("RGB")
        question = item["question"]
        
        # Extract ground truth annotations
        valid_human_answers = [ans["answer"].lower().strip() for ans in item["answers"]]
        best_ground_truth = max(set(valid_human_answers), key=valid_human_answers.count)
        
        # Determine question category
        if best_ground_truth in ["yes", "no"]:
            q_type = "yes/no"
        elif best_ground_truth.isdigit():
            q_type = "number"
        else:
            q_type = "other"

        # Build prompt
        system_prompt = "Answer the question directly and concisely using only a few words."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": f"{system_prompt}\n\nQuestion: {question}"}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=10, temperature=0.01, do_sample=False)
            
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        raw_answer = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
        ai_answer = raw_answer.strip().lower().replace('"', '').replace('.', '')

        # Check match against human references
        is_correct = 1 if ai_answer in valid_human_answers else 0
        
        total_count += 1
        correct_count += is_correct
        cat_stats[q_type]["total"] += 1
        cat_stats[q_type]["correct"] += is_correct

        # If question is Yes/No, log for confusion matrix
        if q_type == "yes/no":
            t_bin = 1 if "yes" in best_ground_truth else 0
            p_bin = 1 if "yes" in ai_answer else 0
            yn_true.append(t_bin)
            yn_pred.append(p_bin)

        # Auto-save row to log CSV
        row_df = pd.DataFrame([{
            "Index": index + 1,
            "Question": question,
            "Ground_Truth": best_ground_truth,
            "AI_Prediction": ai_answer,
            "Is_Correct": is_correct,
            "Question_Type": q_type
        }])
        row_df.to_csv(LOG_CSV, mode='a', header=False, index=False)

        # Update progress bar
        acc = (correct_count / total_count) * 100
        progress_bar.set_postfix({"Acc": f"{acc:.1f}%"})
        progress_bar.update(1)

    except Exception as e:
        tqdm.write(f"[WARNING] Skipping sample {index + 1} due to error: {e}")
        
    finally:
        # VRAM Cleanup per iteration
        if 'inputs' in locals(): del inputs
        if 'generated_ids' in locals(): del generated_ids
        if 'generated_ids_trimmed' in locals(): del generated_ids_trimmed
        if 'image_inputs' in locals(): del image_inputs
        if 'video_inputs' in locals(): del video_inputs
        if 'pil_image' in locals(): del pil_image
        torch.cuda.empty_cache()

progress_bar.close()

# ==========================================
# 5. GENERATE FACULTY REPORTS & GRAPHICS
# ==========================================
print("\n" + "=" * 65)
print("                A.R.G.U.S. EVALUATION METRICS TABLE             ")
print("=" * 65)

overall_acc = (correct_count / total_count) * 100 if total_count > 0 else 0
yn_acc = (cat_stats["yes/no"]["correct"] / cat_stats["yes/no"]["total"] * 100) if cat_stats["yes/no"]["total"] > 0 else 0
num_acc = (cat_stats["number"]["correct"] / cat_stats["number"]["total"] * 100) if cat_stats["number"]["total"] > 0 else 0
oth_acc = (cat_stats["other"]["correct"] / cat_stats["other"]["total"] * 100) if cat_stats["other"]["total"] > 0 else 0

summary_data = {
    "Category": ["Overall Dataset", "Yes/No Subset", "Number/Count Subset", "Other/Open Subset"],
    "Evaluated Samples": [total_count, cat_stats["yes/no"]["total"], cat_stats["number"]["total"], cat_stats["other"]["total"]],
    "Accuracy (%)": [round(overall_acc, 2), round(yn_acc, 2), round(num_acc, 2), round(oth_acc, 2)]
}
df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))

# Yes/No Binary Classification Report
if len(yn_true) > 0:
    print("\n" + "=" * 65)
    print("           YES/NO SUBSET CLASSIFICATION REPORT (PRECISION/RECALL) ")
    print("=" * 65)
    report_dict = classification_report(yn_true, yn_pred, target_names=["Negative (No)", "Positive (Yes)"], output_dict=True)
    df_metrics = pd.DataFrame(report_dict).transpose()
    print(df_metrics.round(3).to_string())

    conf_matrix = confusion_matrix(yn_true, yn_pred)
    print("\n" + "=" * 65)
    print("                         CONFUSION MATRIX                         ")
    print("=" * 65)
    print(f"True Negatives  (TN): {conf_matrix[0][0]:<5} | False Positives (FP): {conf_matrix[0][1]:<5}")
    print(f"False Negatives (FN): {conf_matrix[1][0]:<5} | True Positives  (TP): {conf_matrix[1][1]:<5}")
    print("=" * 65)

    # Save summary & metrics CSV
    df_metrics.to_csv(METRICS_CSV)
    print(f"\n[SYSTEM] Academic metrics saved to '{METRICS_CSV}'")

    # Generate Confusion Matrix Heatmap
    plt.figure(figsize=(7, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title(f"A.R.G.U.S. VQA Confusion Matrix (Yes/No Acc: {yn_acc:.1f}%)")
    plt.ylabel('Ground Truth')
    plt.xlabel('AI Prediction')
    plt.tight_layout()
    plt.savefig(HEATMAP_PNG, dpi=300)
    print(f"[SYSTEM] Confusion matrix plot saved to '{HEATMAP_PNG}'")

print(f"[SYSTEM] Complete per-sample log saved to '{LOG_CSV}'")