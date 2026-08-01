import torch
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

print("\n[SYSTEM] Booting Automated Benchmark Evaluator...")

# ==========================================
# 1. LOAD THE QWEN MODEL (4-bit for 16GB VRAM)
# ==========================================
print("[SYSTEM] Loading Qwen2-VL 2B...")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# NOTE: If you already fine-tuned your model using the previous steps, 
# you can load your custom adapter here instead of the base model!
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", 
    quantization_config=bnb_config,
    device_map="cuda"
)

# ==========================================
# 2. STREAM THE MASSIVE DATASET
# ==========================================
print("[SYSTEM] Connecting to Hugging Face Cloud...")
# We use streaming=True so it doesn't download gigabytes to your hard drive!
# We use the 'adversarial' subset of POPE which is the hardest test for the AI.
dataset = load_dataset("lmms-lab/POPE", name="Full", split="adversarial", streaming=True)

# Define how many samples you want to test today (e.g., 500 for a solid evaluation graph)
TARGET_EVAL_COUNT = 500 

y_true = []
y_pred = []

print(f"\n[SYSTEM] Starting Evaluation Loop for {TARGET_EVAL_COUNT} images...\n")
print("-" * 50)

# ==========================================
# 3. THE EVALUATION LOOP
# ==========================================
for index, item in enumerate(dataset):
    if index >= TARGET_EVAL_COUNT:
        break

    # The POPE dataset provides the image, the question, and the exact "yes" or "no" answer
    pil_image = item["image"].convert("RGB")
    question = item["question"]
    ground_truth = item["answer"].strip().lower()

    # Force the AI to answer with ONLY 'yes' or 'no'
    system_prompt = "You are a highly observant AI. Answer the question with only a single word: 'Yes' or 'No'."
    
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

    # Generate answer
    generated_ids = model.generate(**inputs, max_new_tokens=5, temperature=0.01, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    raw_answer = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    
    # Parse the AI's answer
    ai_answer = raw_answer.replace('"', '').replace('.', '').strip().lower()

    # Convert to binary (1 for Yes, 0 for No)
    true_binary = 1 if "yes" in ground_truth else 0
    pred_binary = 1 if "yes" in ai_answer else 0

    y_true.append(true_binary)
    y_pred.append(pred_binary)

    print(f"Test [{index + 1}/{TARGET_EVAL_COUNT}] | Q: {question} | Truth: {ground_truth.upper()} | AI: {ai_answer.upper()}")

    # Free VRAM
    del inputs, generated_ids, generated_ids_trimmed, pil_image
    torch.cuda.empty_cache()

# ==========================================
# 4. GENERATE FACULTY REPORTS
# ==========================================
print("\n" + "=" * 60)
print("              A.R.G.U.S. EVALUATION METRICS TABLE           ")
print("=" * 60)

# Scikit-Learn generates all the academic metrics for you instantly
report_dict = classification_report(y_true, y_pred, target_names=["Negative (No)", "Positive (Yes)"], output_dict=True)
df_metrics = pd.DataFrame(report_dict).transpose()

print(df_metrics.round(3).to_string())

conf_matrix = confusion_matrix(y_true, y_pred)

print("\n" + "=" * 60)
print("                   CONFUSION MATRIX                         ")
print("=" * 60)
print(f"True Negatives  (TN): {conf_matrix[0][0]:<5} | False Positives (FP): {conf_matrix[0][1]:<5}")
print(f"False Negatives (FN): {conf_matrix[1][0]:<5} | True Positives  (TP): {conf_matrix[1][1]:<5}")
print("=" * 60)

# Save to CSV so you can put it in your presentation or research paper!
df_metrics.to_csv("argus_evaluation_metrics.csv")
print("\n[SYSTEM] Academic evaluation table saved as 'argus_evaluation_metrics.csv'")