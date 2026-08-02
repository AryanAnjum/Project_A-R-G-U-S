import torch
import os
import time
from datasets import load_dataset
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
# 1. 100GB CONFIGURATION
# ==========================================
LOCAL_DATASET_DIR = "../LLaVA-OneVision-Data" 
OUTPUT_DIR = "./qwen2-vl-argus-100gb-checkpoints"
DATASET_CONFIG = "sharegpt4v(llava)"

# 100GB is roughly 350,000 samples. 
MAX_SAMPLES = 350000       

# Save a checkpoint every 1,000 steps (roughly every ~45 mins)
SAVE_EVERY_STEPS = 1000    

print("\n[SYSTEM] Booting 100GB Production QLoRA Pipeline...")

# ==========================================
# 2. CHECKPOINT RESUME DETECTOR
# ==========================================
last_checkpoint = None
if os.path.exists(OUTPUT_DIR):
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
    if last_checkpoint is not None:
        print("=" * 65)
        print(f"🔄 [RESUME DETECTED] Resuming 100GB run from: {last_checkpoint}")
        print("=" * 65)

# ==========================================
# 3. INDUSTRIAL DATASET PROCESSING (.map)
# ==========================================
# We use separate functions so Hugging Face can process the 100GB on the hard drive
# without loading it all into your System RAM.
def filter_invalid(sample):
    return "image" in sample and sample["image"] is not None

def format_sample(sample):
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
            "content": [{"type": "text", "text": assistant_text}]
        }
    ]
    return {"messages": messages}

print(f"\n[SYSTEM] Accessing Local 600GB Repository...")
raw_dataset = load_dataset(LOCAL_DATASET_DIR, DATASET_CONFIG, split="train")

print(f"[SYSTEM] Slicing out {MAX_SAMPLES} samples (~100GB)...")
sliced_dataset = raw_dataset.select(range(min(MAX_SAMPLES, len(raw_dataset))))

print("[SYSTEM] Filtering and formatting data directly on disk (This takes a few minutes)...")
# .filter and .map write cached files to disk, saving your System RAM
valid_dataset = sliced_dataset.filter(filter_invalid)
train_dataset = valid_dataset.map(format_sample, remove_columns=valid_dataset.column_names)

print(f"[SYSTEM] 100GB Dataset locked! Final training size: {len(train_dataset)} samples.")

# ==========================================
# 4. REAL-TIME PROGRESS TRACKER
# ==========================================
class ProgressTrackerCallback(TrainerCallback):
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
            
            samples_per_sec = current_sample / max(elapsed, 1.0)
            remaining_samples = max(0, self.total_samples - current_sample)
            eta_sec = remaining_samples / max(samples_per_sec, 0.001)

            elapsed_str = time.strftime("%Hh:%Mm:%Ss", time.gmtime(elapsed))
            eta_str = time.strftime("%Hh:%Mm:%Ss", time.gmtime(eta_sec))

            print(f"📍 [PROGRESS] Step {state.global_step}/{state.max_steps} "
                  f"| Image ~{current_sample}/{self.total_samples} "
                  f"| Elapsed: {elapsed_str} "
                  f"| ETA: {eta_str}")

# ==========================================
# 5. DATA COLLATOR
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
# 6. MODEL & LORA INJECTION
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

# ==========================================
# 7. 100GB TRAINING EXECUTION
# ==========================================
batch_size = 2
grad_accum = 4

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    learning_rate=2e-4,
    
    save_strategy="steps",
    save_steps=SAVE_EVERY_STEPS,
    save_total_limit=3,                
    
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

print("\n🔥 [SYSTEM] LAUNCHING MASSIVE TRAINING ENGINE 🔥")
trainer.train(resume_from_checkpoint=last_checkpoint)

# Save final adapter model
final_adapter_dir = os.path.join(OUTPUT_DIR, "final_adapter")
trainer.model.save_pretrained(final_adapter_dir)
processor.save_pretrained(final_adapter_dir)

print("\n" + "=" * 65)
print(f"🎉 [100GB TRAINING COMPLETE] Final LoRA adapter saved to:\n   -> {final_adapter_dir}")
print("=" * 65)