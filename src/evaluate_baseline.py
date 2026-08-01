import torch
import json
import time
import os
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ==========================================
# 1. LOAD THE DATASET
# ==========================================
DATASET_PATH = "eval_dataset.json"

if not os.path.exists(DATASET_PATH):
    print(f"[ERROR] Could not find {DATASET_PATH}. Please create it first.")
    exit()

with open(DATASET_PATH, "r") as f:
    eval_data = json.load(f)

print(f"[SYSTEM] Loaded {len(eval_data)} test cases from {DATASET_PATH}.")

# ==========================================
# 2. BOOT THE AI (Qwen2-VL-2B)
# ==========================================
print("[SYSTEM] Loading Qwen2-VL 2B for Evaluation...")
spatial_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", 
    torch_dtype=torch.bfloat16, 
    device_map="cuda"
)
spatial_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# ==========================================
# 3. METRICS TRACKERS
# ==========================================
total_tests = len(eval_data)
total_inference_time = 0.0
results_log = []

# Lists to hold binary representations for sklearn metrics
y_true = []
y_pred = []

system_prompt = (
    "You are a highly observant AI. Look very closely at the fine details. "
    "Answer the user's question directly and concisely."
)

# ==========================================
# 4. THE EVALUATION LOOP
# ==========================================
print("\n[SYSTEM] Starting Evaluation Loop...\n")
print("-" * 50)

for index, item in enumerate(eval_data):
    image_path = item["image_path"]
    question = item["question"]
    ground_truth = str(item["ground_truth"]).lower().strip()

    if not os.path.exists(image_path):
        print(f"Skipping Test {index + 1}: Image {image_path} not found.")
        continue

    pil_image = Image.open(image_path).convert("RGB")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": f"{system_prompt}\n\nQuestion: {question}"}
            ]
        }
    ]

    # Pre-process data
    text = spatial_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = spatial_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    # Start Inference Timer
    start_time = time.time()
    
    generated_ids = spatial_model.generate(**inputs, max_new_tokens=40, temperature=0.1, do_sample=False)
    
    # Stop Inference Timer
    end_time = time.time()
    inference_time = end_time - start_time
    total_inference_time += inference_time

    # Decode answer
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    raw_answer = spatial_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    # Clean up string for grading
    ai_answer = raw_answer.replace('"', '').strip().lower()

    # --- GRADING & BINARY MAPPING LOGIC ---
    # Determine correctness (1 = Correct/Positive, 0 = Incorrect/Negative)
    is_exact = (ground_truth == ai_answer)
    is_partial = (ground_truth in ai_answer)
    
    # For Yes/No ground truths, map directly to binary (1 = Yes, 0 = No)
    if ground_truth in ["yes", "no"]:
        gt_binary = 1 if ground_truth == "yes" else 0
        pred_binary = 1 if "yes" in ai_answer else 0
    else:
        # For general open questions, treat target condition as Correct (1) vs Incorrect (0)
        gt_binary = 1
        pred_binary = 1 if is_partial else 0

    y_true.append(gt_binary)
    y_pred.append(pred_binary)

    # Log individual test result
    print(f"Test {index + 1}/{total_tests}")
    print(f"Image:  {image_path}")
    print(f"Q:      {question}")
    print(f"Truth:  {ground_truth}")
    print(f"AI:     {ai_answer}")
    print(f"Time:   {inference_time:.2f}s | Pass: {is_partial}")
    print("-" * 50)

    results_log.append({
        "test_id": index + 1,
        "image": image_path,
        "question": question,
        "ground_truth": ground_truth,
        "ai_answer": ai_answer,
        "inference_time_sec": round(inference_time, 2),
        "exact_match": is_exact,
        "partial_match": is_partial
    })

    # Free VRAM
    del inputs, generated_ids, generated_ids_trimmed, pil_image
    torch.cuda.empty_cache()

# ==========================================
# 5. CALCULATE STATISTICAL & ACADEMIC METRICS
# ==========================================
avg_inference = total_inference_time / total_tests if total_tests > 0 else 0

# Compute sklearn classification report & confusion matrix
report_dict = classification_report(
    y_true, 
    y_pred, 
    target_names=["Negative / Fail (0)", "Positive / Pass (1)"], 
    output_dict=True,
    zero_division=0
)
df_report = pd.DataFrame(report_dict).transpose()

conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = conf_matrix.ravel() if conf_matrix.size == 4 else (0, 0, 0, 0)

# ==========================================
# 6. OUTPUT SUMMARY TABLES
# ==========================================
print("\n" + "=" * 65)
print("             A.R.G.U.S. ACADEMIC EVALUATION TABLE           ")
print("=" * 65)
print(df_report.round(3).to_string())

print("\n" + "=" * 65)
print("                      CONFUSION MATRIX                      ")
print("=" * 65)
print(f" True Negatives  (TN): {tn:<5} | False Positives (FP): {fp:<5}")
print(f" False Negatives (FN): {fn:<5} | True Positives  (TP): {tp:<5}")
print("-" * 65)
print(f" Average Inference Latency: {avg_inference:.2f} seconds / sample")
print("=" * 65)

# Save JSON and CSV reports for presentation/paper insertion
report_summary = {
    "metrics": {
        "total_tests": total_tests,
        "average_inference_time_sec": round(avg_inference, 2),
        "accuracy": round(report_dict["accuracy"], 3),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    },
    "classification_report": report_dict,
    "detailed_results": results_log
}

with open("baseline_report.json", "w") as f:
    json.dump(report_summary, f, indent=4)

df_report.to_csv("baseline_metrics_table.csv")

print("\n[SYSTEM] Reports successfully exported to 'baseline_report.json' and 'baseline_metrics_table.csv'")