# Fine-tuning BLOOMZ-560m for English-Chinese translation with LoRA, tracking losses and applied layers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch
import torch.nn.functional as F
import math
import os
import csv
from datetime import datetime

# 1. Load dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files="C:/Users/gerri/Documents/Documenten/jaar3/thesis/exp2/dataset_gpt/zho-eng_8_renamed.json", split="train")

# 2. Load tokenizer and base model
checkpoint = "bigscience/bloomz-560m"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"

base_model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    torch_dtype=torch.float32,
    trust_remote_code=True
).to(device)

# 3. Add LoRA adapter
target_modules = [
    # attention layers
    "query_key_value",  # QKV projection in attention
    "dense",  # Output projection in attention
    # feed-forward layers
    "mlp.dense_h_to_4h",  # Hidden to intermediate in MLP
    "mlp.dense_4h_to_h",  # Intermediate to hidden in MLP
]

# 2. CONSERVATIVE PEFT CONFIG (Small Dataset)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,  
    lora_alpha=16, 
    lora_dropout=0.05,
    target_modules=target_modules,
    bias="none",
)

model = get_peft_model(base_model, peft_config)

model.config.pad_token_id = tokenizer.pad_token_id

# Inspect trainable parameters and LoRA-applied modules
print("LoRA applied modules and trainable parameters:")
model.print_trainable_parameters()
print("\nLoRA layers applied:")
for name, module in model.named_modules():
    if "lora" in name.lower():
        print(f"{name} -> {module.__class__.__name__}")

# 4. Tokenization

def tokenize_function(example):
    full_prompt = example["prompt"]
    prompt_tokens = tokenizer(full_prompt, add_special_tokens=True, truncation=True, max_length=200, return_tensors="pt")
    completion_tokens = tokenizer(example["completion"], add_special_tokens=False, truncation=True, max_length=56, return_tensors="pt")

    input_ids = torch.cat([prompt_tokens["input_ids"][0], completion_tokens["input_ids"][0]])
    attention_mask = torch.cat([prompt_tokens["attention_mask"][0], completion_tokens["attention_mask"][0]])
    labels = torch.tensor([-100] * len(prompt_tokens["input_ids"][0]) + completion_tokens["input_ids"][0].tolist())

    max_length = 256
    padding_length = max_length - len(input_ids)
    if padding_length > 0:
        input_ids = torch.cat([input_ids, torch.tensor([tokenizer.pad_token_id] * padding_length)])
        attention_mask = torch.cat([attention_mask, torch.tensor([0] * padding_length)])
        labels = torch.cat([labels, torch.tensor([-100] * padding_length)])
    else:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "labels": labels.tolist()
    }

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=False)

# 5. Split
split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split["train"]
eval_dataset = split["test"]

# 6. Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 7. Training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_lora_model_zhoeng_medium",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    save_total_limit=1,
    save_steps=20,
    logging_steps=5,
    logging_dir="./logs_lora",
    eval_strategy="steps",
    eval_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_safetensors=False,
    learning_rate=4e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    optim="adamw_torch",
    fp16=torch.cuda.is_available(),
)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.05)],
)

# 10. Train
print("Starting training...")
trainer.train()

# 11. Save
model.save_pretrained("./finetuned_lora_model_zhoeng_medium")
tokenizer.save_pretrained("./finetuned_lora_model_zhoeng_medium")
print("Model saved!")

# 12. Log final metrics
log_path = "training_log.csv"
row = {
    "timestamp": datetime.now().isoformat(),
    "learning_rate": training_args.learning_rate,
    "lora_r": peft_config.r,
    "epochs": training_args.num_train_epochs,
    "final_train_loss": trainer.state.log_history[-1].get("loss", None),
}
write_header = not os.path.exists(log_path)
with open(log_path, "a", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=row.keys())
    if write_header:
        writer.writeheader()
    writer.writerow(row)
print("Training log updated.")

# 12. Load for inference (combining base model with adapter)
def load_finetuned_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.float32,
        device_map="auto", 
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, "./finetuned_lora_model_zhoeng_medium")
    model = model.merge_and_unload()  # Merge adapter weights with base model for faster inference
    return model

# 13. Compute log probability and perplexity
def compute_logprob_perplexity(model, full_sequence_ids, generated_ids):
    """
    Computes total log-prob and perplexity of target given prompt + generated
    full_sequence_ids: prompt tokens + generated tokens
    generated_ids: generated tokens only
    """
    with torch.no_grad():
        outputs = model(full_sequence_ids.unsqueeze(0))
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]

    # logit i predicts token i+1
    gen_len = generated_ids.shape[0]
    prediction_logits = logits[0, -gen_len - 1 : -1, :]  # [gen_len, vocab]
    target_ids = generated_ids  # [gen_len]

    # turn to probs
    log_probs = F.log_softmax(prediction_logits, dim=-1)
    selected_log_probs = log_probs[range(gen_len), target_ids]
    total_log_prob = selected_log_probs.sum().item()
    avg_log_prob = total_log_prob / gen_len
    perplexity = math.exp(-avg_log_prob)

    return total_log_prob, perplexity

# 14. Improved generation function
def generate_translation(model, tokenizer, prompt, strategy="greedy"):
    """
    Generate a translation based on specific strategy
    """
    # Prepare clear instruction prompt
    full_prompt = f"Translate the following English text to Chinese:\nEnglish: {prompt}\nChinese:"
    
    # Standard settings
    gen_args = {
        "max_new_tokens": 100,
        "return_dict_in_generate": True,
        "output_scores": True
    }
    
    # Pick right strategy
    if strategy == "greedy":
        gen_args.update({"do_sample": False})
    elif strategy == "top-k_big":
        gen_args.update({"do_sample": True, "top_k": 50, "temperature": 0.7})
    elif strategy == "top-k_small":
        gen_args.update({"do_sample": True, "top_k": 10, "temperature": 0.7})
    elif strategy == "top-p_big":
        gen_args.update({"do_sample": True, "top_p": 0.95, "temperature": 0.7})
    elif strategy == "top-p_small":
        gen_args.update({"do_sample": True, "top_p": 0.75, "temperature": 0.7})
    elif strategy == "beam_big":
        gen_args.update({"do_sample": False, "num_beams": 6, "early_stopping": True})
    elif strategy == "beam_small":
        gen_args.update({"do_sample": False, "num_beams": 3, "early_stopping": True})
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Input and output for model
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device if hasattr(model, "device") else device)
    with torch.no_grad():
        output = model.generate(**inputs, **gen_args)

    # Get the generated part and full sequence
    full_sequence_ids = output.sequences[0]
    generated_ids = full_sequence_ids[inputs.input_ids.shape[1]:]
    generated_decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Compute metrics
    log_probs, perplexity = compute_logprob_perplexity(model, full_sequence_ids, generated_ids)

    return generated_decoded.strip(), log_probs, perplexity

import csv
from datetime import datetime

log_path = "training_log.csv"
row = {
    "timestamp": datetime.now().isoformat(),
    "learning_rate": training_args.learning_rate,
    "lora_r": peft_config.r,
    "epochs": training_args.num_train_epochs,
    "final_train_loss": trainer.state.log_history[-1].get("loss", None),
}
write_header = not os.path.exists(log_path)

with open(log_path, "a", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=row.keys())
    if write_header:
        writer.writeheader()
    writer.writerow(row)