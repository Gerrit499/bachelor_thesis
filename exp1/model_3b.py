# 1 Setup
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import math

# Model
checkpoint = "bigscience/bloomz-3b"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, torch_dtype="auto", device_map="auto", trust_remote_code=True
)
model.eval()


# 2 Log prob and perplexity
def compute_logprob_perplexity(full_sequence_ids, generated_ids):
    """
    Computes total log-prob and perplexity of target given  prompt + generated
    full_sequence_ids: prompt tokens + generated tokens
    generated_ids: generated tokens only
    """
    with torch.no_grad():
        outputs = model(full_sequence_ids.unsqueeze(0))
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        # For every token in sequence 1 score per word in vocab

    # Logit i predicts token i+1
    gen_len = generated_ids.shape[0]
    prediction_logits = logits[0, -gen_len - 1 : -1, :]  # [gen_len, vocab]
    target_ids = generated_ids  # [gen_len]

    # Turn to probs
    log_probs = F.log_softmax(prediction_logits, dim=-1)
    selected_log_probs = log_probs[range(gen_len), target_ids] #  Given the token id, get the log prob of that token
    total_log_prob = selected_log_probs.sum().item() # Sum of log probs
    avg_log_prob = total_log_prob / max(gen_len, 1)
    perplexity = math.exp(-avg_log_prob) # Perplexity

    return total_log_prob, perplexity


# 3 Generation Function
def generate_translation(prompt, strategy="greedy"):
    """
    Genrerate a translation based on specific strategy
    """
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
        gen_args.update({"do_sample": True, "top_k": 50})
    elif strategy == "top-k_small":
        gen_args.update({"do_sample": True, "top_k": 10})
    elif strategy == "top-p_big":
        gen_args.update({"do_sample": True, "top_p": 0.95})
    elif strategy == "top-p_small":
        gen_args.update({"do_sample": True, "top_p": 0.75})
    # Early stopping = can stop top rank reaches eos
    elif strategy == "beam_big":
        gen_args.update({"do_sample": False, "num_beams": 6, "early_stopping": True})
    elif strategy == "beam_small":
        gen_args.update({"do_sample": False, "num_beams": 3, "early_stopping": True})
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Input and output for model (model.device = to GPU)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, **gen_args)

    # Decode only the generated part (remove the prompt from the output)
    generated_tokens = output.sequences[0][inputs.input_ids.shape[1]:]  # only the generated part
    generated_decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Generated_tokens = output.sequences[0][inputs.input_ids.shape[1]:]  # only the generated part
    full_sequence_ids = output.sequences[0]
    generated_ids = full_sequence_ids[inputs.input_ids.shape[1]:]

    # Get from 2
    log_probs, perplexity = compute_logprob_perplexity(full_sequence_ids, generated_ids)

    return generated_decoded.strip(), log_probs, perplexity

    # print(f"{strategy} | log_prob: {log_probs:.2f} | avg: {avg_log_prob:.2f} | translation: {decoded}")
    # print(f"OUTPUT: {tokenizer.decode(output.sequences[0], skip_special_tokens=True)}")

    # print("hey")
    # print(f"{strategy} | log_prob: {log_probs:.2f} | avg: {avg_log_prob:.2f} | translation: {decoded}")
    # print(f"OUTPUT: {tokenizer.decode(output.sequences[0], skip_special_tokens=True)}")