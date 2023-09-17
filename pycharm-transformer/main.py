import os
from transformers import AutoModel
from transformers import AutoTokenizer
import sys
import torch

warmup_step = 2

total_step = 4

dir_name = os.path.dirname(__file__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    batch_size = int(sys.argv[1])
except:
    batch_size = 1

seq = ""
for i in range(30):
    seq =  seq + "Using a Transformer network is simple"
seqs = [seq for _ in range(batch_size)]

checkpoint = os.path.join(dir_name, "../bert-base-uncased")
model = AutoModel.from_pretrained(checkpoint).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(seqs, return_tensors="pt").to(device)

print(inputs['input_ids'].shape)

# print(model)

print(batch_size)

for i in range(total_step):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if i >= warmup_step:
        print("*****************************************************************************************")
        # start.record()
    outputs = model(**inputs)
    # if i >= warmup_step:
        # end.record()
        # torch.cuda.synchronize()
        # print("BertModel", start.elapsed_time(end))

# print(outputs["last_hidden_state"].device)


