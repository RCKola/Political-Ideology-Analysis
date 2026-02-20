from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

# 1. Define paths
adapter_path = "data/centerloss_sbert"
base_model_name = "sentence-transformers/all-MiniLM-L6-v2"
output_path = "data/centerloss_sbert_full" 

print(f"Loading base model: {base_model_name}...")

base_model = AutoModel.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)


print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)


print("Merging weights...")
model = model.merge_and_unload()


print(f"Saving full model to {output_path}...")
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

