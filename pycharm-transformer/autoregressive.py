import os
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import sys
import torch

dir_name = os.path.dirname(__file__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    batch_size = int(sys.argv[1])
except:
    batch_size = 1

seq = ""
for i in range(50):
    seq =  seq + "Using a Transformer network is simple"
seqs = [seq for _ in range(batch_size)]

checkpoint = os.path.join(dir_name, "../bert-base-uncased")
model = AutoModelForCausalLM.from_pretrained(checkpoint, is_decoder=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(seqs, return_tensors="pt").to(device)

print(inputs['input_ids'].shape)

# print(model)

print(batch_size)

model.generate(**inputs, max_new_tokens=4, do_sample=True, use_cache=True)



