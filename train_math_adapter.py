import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
DATA_JSON = "math_data/train_samples_all_tuning.json"
IMG_DIR = "math_data/data_images"
OUTPUT_DIR = "qwen2-vl-argus-math-adapter"

# ==========================================
# 2. DATASET PARSER 
# ==========================================
class MathV360KDataset(Dataset):
    def __init__(self, json_path, img_dir):
        print(f"[SYSTEM] Loading dataset from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        self.data = self.data[:100000] 
        self.img_dir = img_dir
        print(f"[SYSTEM] Loaded {len(self.data)} mathematical samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        img_path = os.path.join(self.img_dir, item["image"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"\n[WARNING] Image load failed: {img_path} | Error: {e}")
            image = Image.new('RGB', (224, 224), color='white')

        human_text = item["conversations"][0]["value"].replace("<image>\n", "").replace("\n<image>", "").strip()
        assistant_text = item["conversations"][1]["value"].strip()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": human_text}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_text}
                ]
            }
        ]
        return messages

# ==========================================
# 3. WINDOWS-SAFE DATA COLLATOR CLASS
# ==========================================
# By making this a class, we guarantee the background workers 
# always have access to the processor.
class CustomDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        batch_texts = []
        batch_images = []
        
        for messages in batch:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            image_inputs, _ = process_vision_info(messages)
            batch_texts.append(text)
            batch_images.extend(image_inputs)
            
        inputs = self.processor(
            text=batch_texts,
            images=batch_images,
            padding=True,
            return_tensors="pt"
        )
        
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        for token in ["<|image_pad|>", "<|vision_start|>", "<|vision_end|>"]:
            tok_id = self.processor.tokenizer.convert_tokens_to_ids(token)
            if tok_id is not None:
                labels[labels == tok_id] = -100
                
        inputs["labels"] = labels
        return inputs


# ==========================================
# 4. MAIN EXECUTION BLOCK 
# ==========================================
def main():
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
        if last_checkpoint is not None:
            print(f"\n[SYSTEM RECOVERY] Resuming training automatically from: {last_checkpoint}\n")
        else:
            print("\n[SYSTEM] No previous checkpoints found. Starting fresh training run.\n")

    print("[SYSTEM] Initializing 4-bit Base Qwen2-VL-2B (16GB VRAM Profile)...")

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", 
        min_pixels=256 * 28 * 28, 
        max_pixels=1024 * 28 * 28 
    )

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

    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model_math = get_peft_model(base_model, peft_config)

    dataset = MathV360KDataset(DATA_JSON, IMG_DIR)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,        
        gradient_accumulation_steps=4,        
        dataloader_num_workers=2,             
        learning_rate=2e-5,
        num_train_epochs=1,
        fp16=False,
        bf16=True,                            
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,                       
        save_total_limit=3,                   
        optim="paged_adamw_8bit",             
        remove_unused_columns=False,          
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False} 
    )

    trainer = Trainer(
        model=model_math,
        args=training_args,
        train_dataset=dataset,
        # Handing the processor directly into the class so workers can grab it
        data_collator=CustomDataCollator(processor)
    )

    print("\n[SYSTEM] Commencing A.R.G.U.S. Math Module Training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    

    print(f"\n[SYSTEM] Training Complete! Saving final adapter to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("[SYSTEM] Adapter Saved Successfully.")


if __name__ == "__main__":
    main()