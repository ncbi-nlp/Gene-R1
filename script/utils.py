import torch
import numpy as np

from torch import Tensor
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")


def cos_sim(a: Tensor, b: Tensor):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def similarity_score(reference, hypothesis):
    with torch.no_grad():
        encoded = tokenizer(
            [reference, hypothesis], 
            truncation=True,
            padding=True, 
            return_tensors='pt', 
            max_length=64,
        )
        
        embeds = model(**encoded).last_hidden_state[:, 0, :] 
        score = cos_sim(embeds[0], embeds[1])
        similarity_score = float(score.tolist()[0][0])
                        
    return similarity_score