from unsloth import FastLanguageModel
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
import os


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

print("Model loaded !!!")


alpaca_prompt = """You are an expert at writing business rule code from the instruction given to you. Below is the instruction given write the most accurate business rule code.

### Instruction:
{}

### Input:
{}

### Response:
{}"""




df = pd.read_excel("br_finetune_data.xlsx")



EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

def formatting_prompts_func(df):
    instructions = df["Instruction"].tolist()
    inputs = df["Input"].tolist()
    outputs = df["Output"].tolist()
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# Apply formatting to the DataFrame
formatted_data = formatting_prompts_func(df)

# Create a dataset from the formatted data
dataset = Dataset.from_dict(formatted_data)



from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 200,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)


trainer_stats = trainer.train()
#######################################################################################################

# Some functions to save the model without getting Runtime error of CUDA out of memory error 

# try:
#     model.save_pretrained_gguf("BR_Finetuned", tokenizer=tokenizer, max_shard_size="1GB")  # Adjust shard size as necessary
# except Exception as e :
#     try:
#         print("Error saving using GPU" , e)
#         model = model.to('cpu')
#         # Save the model using the correct tokenizer reference if needed
#         model.save_pretrained_gguf("BR_Finetuned", tokenizer=tokenizer, max_shard_size="1GB")
#     except Exception as e :
#         print("Error saving using CPU also !!!" , e)





###########################################################################################
## Save the LORA adapters created while finetuning the model
# Local saving
model.save_pretrained("BR_Lora_Adapters") 
tokenizer.save_pretrained("BR_Lora_Adapters")


# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving
###########################################################################################


### Save Entire model


# Save to 8bit Q8_0
# if True : model.save_pretrained_gguf("BR_Finetuned", tokenizer=tokenizer, max_shard_size="500MB") # Local saving
if True : model.save_pretrained_gguf("BR_Finetuned", tokenizer=tokenizer, maximum_memory_usage = 0.5)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False: model.push_to_hub_gguf("ssk2315/BR_llama_8bit", tokenizer, token = "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")


# # Save to 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("ssk2315/Business_Rule_Llama", tokenizer, quantization_method = "f16", token = "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")

# # Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")


# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "hf/model", # Change hf to your username!
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "", # Get a token at https://huggingface.co/settings/tokens
    )



if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
################################################################################################################

