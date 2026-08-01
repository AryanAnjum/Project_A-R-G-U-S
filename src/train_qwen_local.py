import torch
import os
import time
import glob
from datasets import load_dataset, Dataset
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    Trainer,
    TrainerCallback
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
LOCAL_DATASET_DIR = "./LLaVA-OneVision-Data"  # Path to your manually downloaded dataset folder
OUTPUT_DIR = "./qwen2-vl-argus-checkpoints"    # Directory where checkpoints will be saved
MAX_SAMPLES = 10000                           # Number of samples to train on
SAVE_EVERY_STEPS = 100                        # Save a checkpoint every N steps

print("\n[SYSTEM] Booting QLoRA Training Pipeline for A.R.G.U.S...")

# ==========================================
# 1. CHECKPOINT RESUME DETECTOR
# ==========================================
last_checkpoint = None
if os.path.exists(OUTPUT_DIR):
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
    if last_checkpoint is not None:
        print("=" * 65)
        print(f"🔄 [RESUME DETECTED] Found existing checkpoint:")
        print(f"   -> {last_checkpoint}")
        print("   Training will resume seamlessly from this step!")
        print("=" * 65)
    else:
        print("[SYSTEM] Output folder exists, but no previous checkpoint was found. Starting fresh.")
else:
    print("[SYSTEM] No existing checkpoint directory found. Starting new training run.")


# ==========================================
# 2. LOCAL DATASET PARSER
# ==========================================
def prepare_llava_sample(sample):
    """Formats raw dataset items into Qwen2-VL conversational structure."""
    try:
        if "image" not in sample or sample["image"] is None:
            return None
        
        conversations = sample["conversations"]
        user_text = conversations[0]["value"].replace("<image>\n", "").replace("\n<image>", "").strip()
        assistant_text = conversations[1]["value"].strip()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": f"You are A.R.G.U.S., a highly observant AI. Answer concisely.\n\nQuestion: {user_text}"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_text}
                ]
            }
        ]
        return {"messages": messages}
    except Exception:
        return None

print(f"\n[SYSTEM] Loading local dataset from: {LOCAL_DATASET_DIR}...")

# Support loading from local folder or arrow/parquet files
if os.path.exists(LOCAL_DATASET_DIR):
    raw_dataset = load_dataset(LOCAL_DATASET_DIR, split="train")
else:
    print(f"[WARNING] Local path '{LOCAL_DATASET_DIR}' not found! Falling back to cached HuggingFace dataset...")
    raw_dataset = load_dataset("lmms-lab/LLaVA-OneVision-Data", split="train")

processed_dataset = []
print("[SYSTEM] Pre-processing samples into Qwen format...")

for i, sample in enumerate(raw_dataset):
    formatted = prepare_llava_sample(sample)
    if formatted is not None:
        processed_dataset.append(formatted)
        
    if len(processed_dataset) % 1000 == 0 and len(processed_dataset) > 0:
        print(f"   -> Processed {len(processed_dataset)} / {MAX_SAMPLES} samples...")
        
    if len(processed_dataset) >= MAX_SAMPLES:
        break

train_dataset = Dataset.from_list(processed_dataset)
print(f"[SYSTEM] Dataset locked with {len(train_dataset)} training samples.")


# ==========================================
# 3. REAL-TIME PROGRESS & ETA CALLBACK
# ==========================================
class ProgressTrackerCallback(TrainerCallback):
    """Calculates exact elapsed time, ETA, and sample count."""
    def __init__(self, total_samples, batch_size, grad_accum):
        self.start_time = None
        self.total_samples = total_samples
        self.effective_batch = batch_size * grad_accum

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0 and state.global_step > 0:
            elapsed = time.time() - self.start_time
            current_sample = min(state.global_step * self.effective_batch, self.total_samples)
            
            # Estimate remaining time
            samples_per_sec = current_sample / max(elapsed, 1.0)
            remaining_samples = max(0, self.total_samples - current_sample)
            eta_sec = remaining_samples / max(samples_per_sec, 0.001)

            # Format times
            elapsed_str = time.strftime("%Hh:%Mm:%Ss", time.gmtime(elapsed))
            eta_str = time.strftime("%Hh:%Mm:%Ss", time.gmtime(eta_sec))

            print(f"📍 [PROGRESS] Step {state.global_step}/{state.max_steps} "
                  f"| Image ~{current_sample}/{self.total_samples} "
                  f"| Elapsed: {elapsed_str} "
                  f"| ETA Remaining: {eta_str}")


# ==========================================
# 4. DATA COLLATOR FOR QWEN2-VL
# ==========================================
class Qwen2VLDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        texts = []
        image_inputs = []
        
        for feature in features:
            messages = feature["messages"]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            
            imgs, _ = process_vision_info(messages)
            if imgs is not None:
                image_inputs.extend(imgs)

        batch = self.processor(
            text=texts,
            images=image_inputs if image_inputs else None,
            padding=True,
            return_tensors="pt"
        )
        
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch


# ==========================================
# 5. MODEL & QUANTIZATION SETUP (4-BIT)
# ==========================================
print("\n[SYSTEM] Loading 4-bit Quantized Qwen2-VL Model...")

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", 
    min_pixels=256*28*28, 
    max_pixels=1024*28*28 
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    quantization_config=bnb_config,
    device_map="cuda"
)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# ==========================================
# 6. TRAINING ARGUMENTS & EXECUTION
# ==========================================
batch_size = 2
grad_accum = 4

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    learning_rate=2e-4,
    
    # Checkpoint settings
    save_strategy="steps",
    save_steps=SAVE_EVERY_STEPS,
    save_total_limit=3,                # Keeps the 3 latest checkpoints to save disk space
    
    # Logging settings
    logging_steps=10,
    optim="paged_adamw_8bit",
    bf16=True,
    fp16=False,
    remove_unused_columns=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=Qwen2VLDataCollator(processor),
    callbacks=[ProgressTrackerCallback(len(train_dataset), batch_size, grad_accum)]
)

print("\n🔥 [SYSTEM] LAUNCHING TRAINING ENGINE 🔥")
if last_checkpoint:
    print(f"[SYSTEM] Resuming directly from checkpoint: {last_checkpoint}")

trainer.train(resume_from_checkpoint=last_checkpoint)

# Save final adapter model
final_adapter_dir = os.path.join(OUTPUT_DIR, "final_adapter")
trainer.model.save_pretrained(final_adapter_dir)
processor.save_pretrained(final_adapter_dir)

print("\n" + "=" * 65)
print(f"🎉 [TRAINING COMPLETE] Final LoRA adapter saved to:\n   -> {final_adapter_dir}")
print("=" * 65)