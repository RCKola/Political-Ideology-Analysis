from sentence_transformers import SentenceTransformer
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
import torch

# 1. Define paths
adapter_path = "data/centerloss_sbert"
base_model_name = "sentence-transformers/all-MiniLM-L6-v2" # Assuming this is your base
output_path = "data/centerloss_sbert_full" # New path for the fixed model

print(f"Loading base model: {base_model_name}...")
# Load the base model using Hugging Face Transformers
base_model = AutoModel.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 2. Load the adapter onto the base model
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)

# 3. Merge the adapter weights into the base model
print("Merging weights...")
model = model.merge_and_unload()

# 4. Save the full model in a format SentenceTransformer accepts
print(f"Saving full model to {output_path}...")
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print("Done! You can now load the model from:", output_path)