# Overview of Gene-R1

**Introduction**

- Gene-R1 is a data-augmented learning framework that equips lightweight and open-source LLMs with step-by-step reasoning capabilities tailored to the gene set analysis task. 
- It has been fine-tuned by ~270K gene sets collected from 16 genomic databases.
- Experimental results demonstrate that Gene-R1 achieves substantial performance gains, matching commercial LLMs.
- For more details, please check out our [paper](https://www.worldscientific.com/doi/abs/10.1142/9789819824755_0035) (PSB, 2026).

**Gene-R1 helps for gene set analysis through fine-tuned small language models (SLMs) that can be locally deployed.** 
The model contains three versions:
- [Gene-R1-8B](https://huggingface.co/ncbi/Gene-R1-8B): A version fine-tuned based on the Llama-3.1-8B-Instruct.
- [Gene-R1-1B](https://huggingface.co/ncbi/Gene-R1-1B): A version fine-tuned based on the Llama-3.2-1B-Instruct.
- [Gene-R1-3B](https://huggingface.co/ncbi/Gene-R1-3B): A version fine-tuned based on the Llama-3.2-3B-Instruct.


# Model Deployment for Private Gene Set Analysis
## Evaluation
Refer to ``` script/eval.py ```

## Finetuning on your own data
Refer to ``` script/sft.py ```, ``` script/grpo.py ```, and ``` script/utils.py ```

## Finetuning using our data
Please email **zhizheng.wang@nih.gov** and **zhiyong.lu@nih.gov** for the training gene-set data.

# The expected output looks like:
```
  Process: Pancreatic development and glucose homeostasis
  
  1. PDX1 is a homeodomain transcription factor involved in the specification of the early pancreatic epithelium and its subsequent differentiation. 
  It activates the transcription of several genes including insulin, somatostatin, glucokinase and glucose transporter type 2. 
  It is essential for maintenance of the normal hormone-producing phenotype in the pancreatic beta-cell. 
  In pancreatic acinar cells, forms a complex with PBX1b and MEIS2b and mediates the activation of the ELA1 enhancer.
  
  2. NKX6-1 is also a transcription factor involved in the development of pancreatic beta-cells during the secondary transition. 
  Together with NKX2-2 and IRX3, controls the generation of motor neurons in the neural tube and belongs to the neural progenitor 
  factors induced by Sonic Hedgehog (SHH) signals.
  
  3.GCG and GLP1, respectively glucagon and glucagon-like peptide 1, are involved in glucose metabolism and homeostasis. 
  GCG raises blood glucose levels by promoting gluconeogenesis and is the counter regulatory hormone of Insulin. 
  GLP1 is a potent stimulator of Glucose-Induced Insulin Secretion (GSIS). Plays roles in gastric motility and suppresses blood glucagon levels. 
  Promotes growth of the intestinal epithelium and pancreatic islet mass both by islet neogenesis and islet cell proliferation.
  
  4. SLC2A2, also known as GLUT2, is a facilitative hexose transporter. In hepatocytes, it mediates bi-directional transport of glucose accross the plasma membranes, 
  while in the pancreatic beta-cell, it is the main transporter responsible for glucose uptake and part of the cell's glucose-sensing mechanism. 
  It is involved in glucose transport in the small intestine and kidney too.
  
  To summarize, the genes in this set are involved in the specification, differentiation, growth and functionality of the pancreas, 
  with a particular emphasis on the pancreatic beta-cell. Particularly, the architecture of the pancreatic islet ensures proper glucose sensing 
  and homeostasis via a number of different hormones and receptors that can elicit both synergistic and antagonistic effects in the pancreas itself and other peripheral tissues.
```

⚠️ **Notice: The outputs sometimes are not following the instruction, you can try again if this case occurs.**

More details of model usage can be referred at our Hugging Face: [HF](https://huggingface.co/ncbi/Gene-R1-8B)

# Acknowledgments

This research was supported in part by the Intramural Research Program of the National Institutes of Health (NIH). 
The contributions of the NIH authors are considered Works of the United States Government. 
The findings and conclusions presented in this paper are those of the authors and do not necessarily reflect the views of the NIH or the U.S. Department of Health and Human Services.

# Disclaimer

These models show the results of research conducted in the Computational Biology Branch, NCBI/NLM. 
The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. 
Individuals should not change their health behavior solely on the basis of information produced on this website. 
NIH does not independently verify the validity or utility of the information produced by this tool. 
If you have questions about the information produced on this website, please see a health care professional.
More information about NCBI's disclaimer policy is available.

# Citation

```bibtext
@inproceedings{wang2025gene,
  title={Gene-R1: Reasoning with Data-Augmented Lightweight LLMs for Gene Set Analysis},
  author={Wang, Zhizheng and Yang, Yifan and Jin, Qiao and Lu, Zhiyong},
  booktitle={Biocomputing 2026: Proceedings of the Pacific Symposium},
  pages={494--507},
  year={2025},
  organization={World Scientific}
}
```
