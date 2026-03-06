'''

'''
from transformers import logging

logging.set_verbosity_error()
import re
import torch
import openai
import numpy as np
import pandas as pd

from scipy import stats
from rouge_score import rouge_scorer
from torch import Tensor
import tqdm

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_path", default=None)
parser.add_argument("-p", "--partition", default=None)
parser.add_argument("-t", "--task", default=None)
args = parser.parse_args()
out_path = f"{args.task}.finetuned.llama3.{args.model_path.split('/')[-1]}.{args.partition}.result.txt"
with open(out_path, 'w') as f:
  pass
model_path = args.model_path

tokenizer_test = AutoTokenizer.from_pretrained(model_path, token = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx') # Your access key of hugging face
model_test = AutoModelForCausalLM.from_pretrained(
    model_path, device_map='auto', token = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'  # Your access key of hugging face
)

def complete_chat(
                system,
                prompt,
                model,
                tokenizer
):
    model.generation_config.do_sample=False
    tokenized_chat = tokenizer('#SYSTEM: \n'+ system + '#USER: \n'+ prompt+' #Assistant: \n', return_tensors="pt").input_ids.to(model.device)
    # input_length = len(tokenizer.decode(tokenized_chat[0]))
    outputs = model.generate(tokenized_chat, max_new_tokens=4000, temperature = 0) 
    return tokenizer.decode(outputs[0])

##################################
# Configure fine-tuned Llama model
##################################

system = "You are an efficient and insightful assistant to a molecular biologist."
baseline = lambda genes: f"""
Write a critical analysis of the biological processes performed by this system of interacting proteins.
Base your analysis on prior knowledge available in your training data.
After the analysis, propose a brief name for the most prominent biological process performed by the system.
Place the name at the top of the analysis in the format: "Process: <name>".
Be concise. Avoid unnecessary words.
Use plain text only. Do not include format symbols such as asterisks, dashes, or bullets.
Be specific. Avoid overly general statements such as "the proteins are involved in various cellular processes."
Be factual. Do not include editorial opinions or unsupported claims.
For each important point, clearly explain your reasoning and provide supporting information.
For each identified biological function, specify the corresponding gene names.
Here is the gene set: {genes}
"""

## baseline 
def llama(ID, genes):    
    genes = genes.replace("/",",").replace(" ",",")
    #print(f"\n{ID}")
    #print(genes)
    
    prompt_baseline = baseline(genes)
    # ###############################################################
    # ## Configure the fine-tuned Llama here#########################
    # messages = [
    #     {"role":"system", "content":system},
    #     {"role":"user", "content":prompt_baseline}
    # ]
    # summary = openai.ChatCompletion.create(
    #     engine="gpt-4o",
    #     messages=messages,
    #     temperature=0,
    #     )
    # ################################################################
    summary =complete_chat(system, prompt_baseline, model_test, tokenizer_test)

    # messages.append(summary.choices[0]["message"])
    # summary = summary.choices[0]["message"]["content"]
    with open(out_path,"a") as f_summary:
        f_summary.write(summary+"\n")
        f_summary.write(ID+"\n")
        f_summary.write("//\n")
    #print("=====Summary=====")
    #print(summary)
    
def rouge(reference, generation): 
	metrics = ["rougeL", "rouge1", "rouge2"]
	metric2results = {metric: [] for metric in metrics}
	scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)

	for ref, hyp in zip (reference, generation):
		scores_gpt = scorer.score(ref, hyp)
		for metric in metrics:
			metric2results[metric].append(scores_gpt[metric].fmeasure)

	for metric in metrics:
		results = metric2results[metric]
		confidence_level = 0.95
		
		mean = np.average(results)
		stderror = stats.sem(np.asarray(results))
		std = np.std(np.asarray(results))
		degrees_freedom = len(results) - 1
		confidence_interval = stats.t.interval(confidence_level, degrees_freedom, loc=mean, scale=stderror)
			
		print(f"The mean value of {metric} is {mean} \nstandard error is {stderror} \nstandard deviation is {std} \n95% confidence level is {confidence_interval}")
		print("==============================================================\n")
		f = open(f"NeST.Rouge.FineTuned.Llama.{args.model_path.split('/')[-1]}.txt","a")
		f.write(metric + ":" + str(sum(results) / len(results)) + "\n")

def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
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


def similarity(reference, generation):
    scores = []
    for ref, hyp in zip(reference, generation):
        with torch.no_grad():
            encoded_agent = tokenizer(
                [ref, hyp], 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=64,
            )
            embeds = model(**encoded_agent).last_hidden_state[:, 0, :]
            score = cos_sim(embeds[0], embeds[1])
            scores.append(score.tolist()[0])
            
    print(f"the average similarity score obtained by MedCPT is {np.average(scores)}")
    np.savetxt("NeST.Similarity.FineTuned.Llama.txt", np.asarray(scores), fmt="%s", delimiter="\t", newline="\n")  
        
def process_text(text: str) -> list:
    pattern = r'\([^)]*\)'
    segments = text.split('//')
    cleaned_segments = []
    for segment in segments:
        cleaned_segment = ''.join(char for char in segment)
        cleaned_segment = re.sub(pattern, '', cleaned_segment)
        cleaned_segment = cleaned_segment.replace('/', ' ').replace(","," ").replace("\"","").replace("-", " ").strip()
        if cleaned_segment:
            cleaned_segments.append(cleaned_segment)

    return cleaned_segments

            
if __name__ == "__main__":
    ref, hyp = [], []
    if args.task == 'GO':
        data = pd.read_csv("GO_terms.csv", header=0, index_col=None)
    elif args.task == 'CC':
      data = pd.read_csv("CC_test.csv", header=0, index_col=None)
    elif args.task == 'MF':
      data = pd.read_csv("MF_test.csv", header=0, index_col=None)
    elif args.task == 'msig':
      data = pd.read_csv("MsigDB.csv", header=0, index_col=None)
    else:
        data = pd.read_table("NeST.tsv", header=0, index_col=None)
    

    partition = int(args.partition)
    partition_length = int(len(data)/8)
    start = 0+partition*partition_length
    end = 0+(partition+1)*partition_length
    if partition == 7:
        end = len(data)
    data = data[start:end]
    print(f"Partition {partition}: {start}:{end}")
    for ID, genes in tqdm.tqdm(zip(data["ID"], data["Genes"]), total = len(data["ID"]), position=int(args.partition), desc=f"part{args.partition}"):
        # ref.append(label)
        llama(ID, genes)
        
    outcome = ""
    with open (out_path, "r") as llamafile:
        for line in llamafile.readlines():
            outcome += line
    llama_text = process_text(outcome)
    
    # for text in llama_text:
    #     seg = text.split("#Assistant: \nProcess")[1].split("\n") ## This might need to have a modification according to the output
    #     if len(seg) > 1:
    #         hyp.append(seg[0].split(": ")[1])
    #     else:
    #         hyp.append("None")
            
    # rouge(ref, hyp)
    # similarity(ref, hyp)
        
    print("===Finished!===")
    
    
###  The expected output likes followings:
    
# Process: Pancreatic development and glucose homeostasis

# 1. PDX1 is a homeodomain transcription factor involved in the specification of the early pancreatic epithelium and its subsequent differentiation. 
# It activates the transcription of several genes including insulin, somatostatin, glucokinase and glucose transporter type 2. 
# It is essential for maintenance of the normal hormone-producing phenotype in the pancreatic beta-cell. 
# In pancreatic acinar cells, forms a complex with PBX1b and MEIS2b and mediates the activation of the ELA1 enhancer.

# 2. NKX6-1 is also a transcription factor involved in the development of pancreatic beta-cells during the secondary transition. 
# Together with NKX2-2 and IRX3, controls the generation of motor neurons in the neural tube and belongs to the neural progenitor 
# factors induced by Sonic Hedgehog (SHH) signals.

# 3.GCG and GLP1, respectively glucagon and glucagon-like peptide 1, are involved in glucose metabolism and homeostasis. 
# GCG raises blood glucose levels by promoting gluconeogenesis and is the counter regulatory hormone of Insulin. 
# GLP1 is a potent stimulator of Glucose-Induced Insulin Secretion (GSIS). Plays roles in gastric motility and suppresses blood glucagon levels. 
# Promotes growth of the intestinal epithelium and pancreatic islet mass both by islet neogenesis and islet cell proliferation.

# 4. SLC2A2, also known as GLUT2, is a facilitative hexose transporter. In hepatocytes, it mediates bi-directional transport of glucose accross the plasma membranes, 
# while in the pancreatic beta-cell, it is the main transporter responsible for glucose uptake and part of the cell's glucose-sensing mechanism. 
# It is involved in glucose transport in the small intestine and kidney too.

# To summarize, the genes in this set are involved in the specification, differentiation, growth and functionality of the pancreas, 
# with a particular emphasis on the pancreatic beta-cell. Particularly, the architecture of the pancreatic islet ensures proper glucose sensing 
# and homeostasis via a number of different hormones and receptors that can elicit both synergistic and antagonistic effects in the pancreas itself and other peripheral tissues.
    
