from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

model_path = "./mymodel/Llama-2-7b-chat-hf"
if not os.path.exists(model_path):
    os.makedirs(model_path)

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

