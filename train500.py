import torch
import os
import time
import copy
import io
from PIL import Image
from datasets import load_dataset, interleave_datasets
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor, 
    TrainingArguments, 
    Trainer,
    TrainerCallback
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info

# ==========================================
# 1. 500GB 20-SUBSET CONFIGURATION
# ==========================================
LOCAL_DATASET_DIR = "../LLaVA-OneVision-Data" 
OUTPUT_DIR = "./qwen2-vl-argus-500gb-checkpoints"

# Add all your targeted subset names to this array
SUBSETS = [
    "sharegpt4o", "mathqa", "scienceqa", "docvqa", "chartqa",
    "textvqa", "ocrvqa", "ai2d", "geometry3k", "clevr",
    "vqav2", "vizwiz", "infographicvqa", "gqa", "websight",
    "synthdog", "mathvista", "theoremqa", "scibench", "coco"
]

MAX_TRAINING_STEPS = 50000       
SAVE_EVERY_STEPS = 1000    

print(f"\n[SYSTEM] Booting 500GB Iterable Pipeline with {len(SUBSETS)} Subsets...")

# ==========================================
# 2. CHECKPOINT RESUME DETECTOR
# ==========================================
last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
if last_checkpoint:
    print(f"[SYSTEM] Resuming from fault-tolerant checkpoint: {last_checkpoint}")
else:
    print("[SYSTEM] No checkpoint found. Starting fresh training loop.")

# ==========================================
# 3. DYNAMIC 20-STREAM MULTIPLEXING
# ==========================================
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
                {"type": "image", "image": "<PLACEHOLDER>"},
                {"type": "text", "text": f"You are A.R.G.U.S., a highly observant AI. Answer concisely.\n\nQuestion: {user_text}"}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_text}]
        }
    ]
    return {"messages": messages}

print(f"\n[SYSTEM] Initializing Zero-RAM SSD Streams...")

# Dynamically generate a streaming pipeline for every subset in your list
dataset_streams = []
for subset in SUBSETS:
    print(f" -> Opening pipeline for: {subset}")
    stream = load_dataset(LOCAL_DATASET_DIR, subset, split="train", streaming=True)
    dataset_streams.append(stream)

# Calculate an equal probability fraction (1 / 20 = 0.05 for each stream)
equal_probabilities = [1.0 / len(SUBSETS)] * len(SUBSETS)

# Multiplex all 20 streams together dynamically
mixed_stream = interleave_datasets(
    dataset_streams, 
    probabilities=equal_probabilities
)

train_dataset = (
    mixed_stream
    .filter(filter_invalid)
    .map(format_sample, remove_columns=["id", "conversations"])
)

print("[SYSTEM] 20-Stream Multiplexer locked. Data will be fetched on demand.")

# ==========================================
# 4. DATA COLLATOR
# ==========================================
class Qwen2VLDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        texts = []
        image_inputs = []
        
        for feature in features:
            raw_img = feature["image"]
            if isinstance(raw_img, dict):
                if "bytes" in raw_img and raw_img["bytes"]:
                    pil_img = Image.open(io.BytesIO(raw_img["bytes"])).convert("RGB")
                elif "path" in raw_img and raw_img["path"]:
                    pil_img = Image.open(raw_img["path"]).convert("RGB")
            else:
                pil_img = raw_img.convert("RGB")

            messages = copy.deepcopy(feature["messages"])
            
            for msg in messages:
                if "content" in msg:
                    for item in msg["content"]:
                        if item.get("type") == "image":
                            item["image"] = pil_img

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            image_inputs.append(pil_img)
            
        batch = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        labels[labels == image_token_id] = -100
        
        batch["labels"] = labels
        return batch

# ==========================================
# 5. MODEL & LORA INJECTION
# ==========================================
print("\n[SYSTEM] Loading Qwen2-VL Model in Native bfloat16...")

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", 
    min_pixels=256*28*28, 
    max_pixels=640*28*28
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

model.enable_input_require_grads()

peft_config = LoraConfig(
    r=32, 
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

# ==========================================
# 6. 500GB TRAINING EXECUTION
# ==========================================
batch_size = 1  
grad_accum = 16 

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    max_steps=MAX_TRAINING_STEPS,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    learning_rate=2e-4,
    
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    save_strategy="steps",
    save_steps=SAVE_EVERY_STEPS,
    save_total_limit=3,                
    
    logging_steps=10,
    optim="paged_adamw_8bit",
    bf16=True,                     
    fp16=False,
    remove_unused_columns=False,
    report_to="none",
    dataloader_num_workers=4, 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=Qwen2VLDataCollator(processor)
)

print("\n[SYSTEM] Initializing 500GB Streaming Training Loop...")
trainer.train(resume_from_checkpoint=last_checkpoint)