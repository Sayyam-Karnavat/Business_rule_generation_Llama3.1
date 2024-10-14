## from pretrained loads the model saved in gguf format 
from unsloth import FastLanguageModel
alpaca_prompt = """You are an expert at writing business rule code from the instruction given to you. Below is the instruction given write the most accurate business rule code.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "BR_Lora_Adapters",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
print("Model Loaded !!!")

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

print("Model optimized via Peft !!!")


FastLanguageModel.for_inference(model) # Enable native 2x faster inference

print("Model loaded for inference ")




def generate_code(instruction):
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            f'{instruction}', # instruction
            "", # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    generated_business_Rule = tokenizer.batch_decode(outputs)
    index_of_response = str(generated_business_Rule).rfind("Response") + len("Response")
    end_of_text = str(generated_business_Rule).rfind
    return str(generated_business_Rule)[index_of_response:].replace("\n" , "").replace("<|end_of_text|>","")

if __name__ == "__main__":
    while True:
        user_instruction = input("Enter the Instruction :-")
        generated_business_rule = generate_code(instruction= user_instruction)
        print("Business rule :-" , generated_business_rule)
