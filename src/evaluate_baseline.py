import torch
import json
import time
import os
from PIL import Image
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
exact_matches = 0
partial_matches = 0
total_inference_time = 0.0
results_log = []

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

    # --- GRADING LOGIC ---
    is_exact = False
    is_partial = False

    if ground_truth == ai_answer:
        exact_matches += 1
        is_exact = True
        is_partial = True
    elif ground_truth in ai_answer:
        partial_matches += 1
        is_partial = True

    # Log it
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
# 5. CALCULATE FINAL SCORES
# ==========================================
avg_inference = total_inference_time / total_tests if total_tests > 0 else 0
exact_accuracy = (exact_matches / total_tests) * 100 if total_tests > 0 else 0
partial_accuracy = ((exact_matches + partial_matches) / total_tests) * 100 if total_tests > 0 else 0

print("\n==========================================")
print("       BASELINE EVALUATION RESULTS        ")
print("==========================================")
print(f"Total Tests Run:    {total_tests}")
print(f"Exact Match Acc:    {exact_accuracy:.1f}%")
print(f"Partial Match Acc:  {partial_accuracy:.1f}%")
print(f"Avg Inference Time: {avg_inference:.2f} seconds")
print("==========================================")

# Save the final report for your faculty
report = {
    "metrics": {
        "total_tests": total_tests,
        "exact_match_accuracy": exact_accuracy,
        "partial_match_accuracy": partial_accuracy,
        "average_inference_time_sec": round(avg_inference, 2)
    },
    "detailed_results": results_log
}

with open("baseline_report.json", "w") as f:
    json.dump(report, f, indent=4)
    
print("\n[SYSTEM] Full evaluation report saved to 'baseline_report.json'")