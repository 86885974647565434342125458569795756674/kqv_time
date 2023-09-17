docker run --privileged -it -p XXXX:22 --name cyy_transformer --gpus '"device=1"'  -v /data/cyy/transformer:/transformer nvcr.io/nvidia/pytorch:23.08-py3

docker exec --privileged -it cyy_transformer /bin/bash 

passwd

apt update

apt install openssh-server

/etc/ssh/sshd_config

PermitRootLogin yes

service ssh start

\# 已有torch

pip install transformers

scp -r bert-base-uncased cyy@172.18.xxxx:/data/cyy/transformer

# 一次迭代

from transformers import AutoModel

BertModel(
  (embeddings): BertEmbeddings(
    (word_embeddings): Embedding(30522, 768, padding_idx=0)
    (position_embeddings): Embedding(512, 768)
    (token_type_embeddings): Embedding(2, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (encoder): BertEncoder(
    (layer): ModuleList(
      (0-11): 12 x BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
  )
  (pooler): BertPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
  )
)



  (encoder): BertEncoder(
    (layer): ModuleList(
      (0-11): 12 x BertLayer(

​      )
​    )
  )



      (0-11): 12 x BertLayer(
        (attention): BertAttention
        (intermediate): BertIntermediate
        (output): BertOutput
      )



        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
        )



/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py

BertModel.forward

BertEncoder.forward

BertLayer.forward

BertAttention.forward

BertSelfAttention.forward

# 增量阶段

from transformers import AutoModelForCausalLM

/usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py

GenerationMixin.generation

GenerationMixin.sample

outputs = self(

transformers\models\bert\modeling_bert.py

BertLMHeadModel.forward

outputs = self.bert(

BertModel.forward
