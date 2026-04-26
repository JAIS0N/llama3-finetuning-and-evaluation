
#savee merged model

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "meta-llama/Llama-3.2-1B-Instruct"
adapter_path = "/content/best_model"   # your trained adapter
save_path = "./merged-model"           # local folder for vLLM

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype="auto",
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)

# Merge LoRA into base model
model = model.merge_and_unload()

# Save merged model
model.save_pretrained(save_path)

# Save tokenizer (VERY IMPORTANT)
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.save_pretrained(save_path)

print(f"Model saved at: {save_path}")
