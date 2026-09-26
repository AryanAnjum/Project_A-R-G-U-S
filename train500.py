import os
import sys
import copy
import io
import torch
from PIL import Image
from datasets import load_dataset, interleave_datasets
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, PeftModel

# ==========================================
# 1. 5-PART SUBSET PARTITIONS (~100GB EACH)
# ==========================================
LOCAL_DATASET_DIR = "../LLaVA-OneVision-Data" 
BASE_OUTPUT_DIR = "./argus_phased_adapters"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# 20 subsets split logically into 5 distinct training phases
DATASET_PARTS = {
    1: ["sharegpt4o", "mathqa", "scienceqa", "docvqa"],
    2: ["chartqa", "textvqa", "ocrvqa", "ai2d"],
    3: ["geometry3k", "clevr", "vqav2", "vizwiz"],
    4: ["infographicvqa", "gqa", "websight", "synthdog"],
    5: ["mathvista", "theoremqa", "scibench", "coco"]
}

# 2,000 steps per part takes roughly 3 to 4 hours on an RTX 5060 Ti
# Total across 5 parts = 10,000 steps (160,000 samples trained)
STEPS_PER_PART = 2000 
BATCH_SIZE = 1
GRAD_ACCUM = 16

# ==========================================
# 2. DATA COLLATOR & STREAMING HELPERS
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
                {"type": "text", "text": f"You are A.R.G.U.S., a sharp multimodal assistant.\n\nQuestion: {user_text}"}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_text}]
        }
    ]
    return {"messages": messages}

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
# 3. INTERACTIVE PART SELECTOR
# ==========================================
def get_user_choice():
    print("\n" + "="*50)
    print("      A.R.G.U.S. PHASED 500GB TRAINING MENU")
    print("="*50)
    for p in range(1, 6):
        path = os.path.join(BASE_OUTPUT_DIR, f"part_{p}_adapter")
        status = "[COMPLETED]" if os.path.exists(path) else "[NOT STARTED]"
        print(f" Part {p}: {DATASET_PARTS[p]} -> {status}")
    print("="*50)
    
    while True:
        choice = input("\nWhich Part do you want to train? (1-5) or 'q' to quit: ").strip().lower()
        if choice == 'q':
            sys.exit(0)
        if choice in ["1", "2", "3", "4", "5"]:
            return int(choice)
        print("Invalid input. Enter a number between 1 and 5.")

# ==========================================
# 4. TRAINING WORKER
# ==========================================
def run_part_training(part_num):
    print(f"\n[SYSTEM] Preparing Environment for Part {part_num}...")
    current_save_path = os.path.join(BASE_OUTPUT_DIR, f"part_{part_num}_adapter")
    subsets_to_train = DATASET_PARTS[part_num]
    
    # 1. Load Data Streams for this Phase
    dataset_streams = []
    for subset in subsets_to_train:
        print(f" -> Connecting to stream: {subset}")
        stream = load_dataset(LOCAL_DATASET_DIR, subset, split="train", streaming=True)
        dataset_streams.append(stream)

    equal_probs = [1.0 / len(subsets_to_train)] * len(subsets_to_train)
    mixed_stream = interleave_datasets(dataset_streams, probabilities=equal_probs)
    train_dataset = (
        mixed_stream
        .filter(filter_invalid)
        .map(format_sample, remove_columns=["id", "conversations"])
    )

    # 2. Load Base Backbone & Processor
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

    # Anti-crash cache setting
    model.config.use_cache = False
    model.enable_input_require_grads()

    # 3. Model Architecture Wiring
    if part_num == 1:
        # Part 1: Initialize fresh LoRA adapters
        print("[SYSTEM] Initializing fresh LoRA weights for Part 1...")
        peft_config = LoraConfig(
            r=32, 
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    else:
        # Subsequent Parts: Load adapter from previous part in TRAINABLE mode
        prev_part = part_num - 1
        prev_adapter_path = os.path.join(BASE_OUTPUT_DIR, f"part_{prev_part}_adapter")
        if not os.path.exists(prev_adapter_path):
            print(f"[ERROR] Cannot start Part {part_num} because previous adapter '{prev_adapter_path}' does not exist!")
            print(f"[ACTION] Please train Part {prev_part} first.")
            return False
            
        print(f"[SYSTEM] Loading and chaining adapter weights from Part {prev_part}...")
        model = PeftModel.from_pretrained(model, prev_adapter_path, is_trainable=True)

    model.base_model.config.use_cache = False

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_OUTPUT_DIR, f"temp_run_part_{part_num}"),
        max_steps=STEPS_PER_PART,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=1e-4 if part_num > 1 else 2e-4, # Slightly lower LR on chained passes
        
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        save_strategy="no", # We save cleanly at the end of the part
        logging_steps=10,
        optim="paged_adamw_8bit",
        bf16=True,                     
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0, # Prevents Windows threading crashes
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=Qwen2VLDataCollator(processor)
    )

    print(f"\n[TRAINING] Starting Part {part_num} ({STEPS_PER_PART} steps)...")
    trainer.train()

    # 5. Clean Save
    print(f"\n[SYSTEM] Part {part_num} complete! Saving adapter to: {current_save_path}")
    model.save_pretrained(current_save_path)
    processor.save_pretrained(current_save_path)
    print(f"[SUCCESS] Part {part_num} successfully archived to disk.")

    # 6. Memory Cleanup
    del model, trainer, processor, train_dataset
    torch.cuda.empty_cache()
    return True

# ==========================================
# 5. MAIN LOOP
# ==========================================
def main():
    while True:
        part_to_train = get_user_choice()
        success = run_part_training(part_to_train)
        
        if not success:
            continue
            
        if part_to_train == 5:
            print("\n" + "="*50)
            print("CONGRATULATIONS! ALL 5 PARTS (500GB) ARE FULLY TRAINED!")
            print(f"Final model adapter is stored at: {BASE_OUTPUT_DIR}/part_5_adapter")
            print("="*50)
            break

        # Prompt to continue or quit
        next_part = part_to_train + 1
        user_response = input(f"\nPart {part_to_train} is finished and saved. Do you want to continue to Part {next_part} right now? (y/n): ").strip().lower()
        
        if user_response != 'y':
            print(f"\n[SYSTEM] Exiting cleanly. When you run this script again, select Part {next_part} to resume.")
            break

if __name__ == "__main__":
    main()