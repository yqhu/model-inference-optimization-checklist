# install torch & transformers
# pip install torch==1.1.0 transformers==3.3.0

import torch
import transformers
from transformers import BertModel, BertTokenizer
import time

model_path = 'bert-base-uncased'
sequence_length = 16
batch_size = 32

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)

seq_len=(sequence_length - tokenizer.num_special_tokens_to_add(pair=False))
dummy_inputs = [[tokenizer.unk_token] * seq_len] * batch_size
inputs = tokenizer(dummy_inputs, is_split_into_words=True, return_tensors='pt')

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Warmup
for _ in range(10):
    _ = model(input_ids,attention_mask)

# Timing
start = time.perf_counter()

for _ in range(100):
    _ = model(input_ids,attention_mask)

print('torch.__version__:', torch.__version__)
print('transformers.__version__:', transformers.__version__)
print('Total time:', time.perf_counter() - start, 'seconds')
