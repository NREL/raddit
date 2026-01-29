from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn.functional as F
from torch import Tensor

from optimum.quanto import QuantizedModelForCausalLM, qint8

import numpy as np
from tqdm import tqdm
import json

import sys
import os

def process_batch(batch_number):
    with open(f"../data/job_strings/job_strings_{batch_number}.json", "r") as f:
        input_texts = json.load(f)

    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral")
    model = AutoModel.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral")
    qmodel = QuantizedModelForCausalLM.quantize(model, weights=qint8, exclude='lm_head')
    del model

    device = torch.device("cuda")
    qmodel = qmodel.to(device)
    qmodel.eval()

    max_length = 2048

    def last_token_pool(last_hidden_states,
                     attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
            
    all_embeddings = []
    num_batches = (len(input_texts) + 1) // 2
    for i in tqdm(range(num_batches)):
        input_text = input_texts[i*2:(i+1)*2]
        if not input_text:
            continue
        batch_dict = tokenizer(input_text, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = qmodel(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.extend(embeddings.cpu().detach().numpy().tolist())

        del batch_dict
        del outputs
        del embeddings

        torch.cuda.empty_cache()
        
    os.makedirs("../data/embeddings", exist_ok=True)
    np.save(f'../data/embeddings/embeddings_{batch_number}.npy', np.array(all_embeddings))
        

def main():
    if len(sys.argv) != 2:
        print("Usage: embed_job_scripts.py <integer>")
        sys.exit(1)
    num = int(sys.argv[1])
    process_batch(num)

if __name__ == "__main__":
    main()