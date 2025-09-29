import re
import torch
import json
from datasets import load_from_disk
import os
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, GenerationConfig

from trl import (
    HfPairwiseJudge,
    LogCompletionsCallback,
    ModelConfig,
    GRPOConfig, 
    GRPOTrainer,
    OpenAIPairwiseJudge,
    PairRMJudge,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from rouge_score import rouge_scorer
from torch import Tensor
score_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
score_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")


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
        encoded = score_tok(
            [reference, hypothesis], 
            truncation=True,
            padding=True, 
            return_tensors='pt', 
            max_length=64,
        )
        
        embeds = score_model(**encoded).last_hidden_state[:, 0, :] 
        score = cos_sim(embeds[0], embeds[1])
        similarity_score = float(score.tolist()[0][0])
                        
    return similarity_score

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def reward_process_match(completions, **kwargs):
    golds = kwargs["process_name"]
    
    rewards = []
    for comp, gold in zip(completions, golds):
        reward = 0.0
        if "Let me perfrom reasoning for the given genes:" in comp:
            comp = comp.split('Let me perfrom reasoning for the given genes:')[-1]
            
        # Check for <think>...</think>
        has_think = bool(re.search(r"<think>.*?</think>", comp, re.DOTALL))
        if has_think:
            reward += 1.0  # +1 if thinking block present
        
        # Check for \nProcess:
        has_process_section = bool(re.search(r"\sProcess:\s*", comp))
        if has_process_section:
            reward += 1 # +1 if Process: section present

        m = re.search(r"\sProcess:\s*([^\n\.\:,]+)", comp)
        extracted = m.group(1).strip() if m else None
        if extracted != None:
            rougel = scorer.score(extracted, gold)["rougeL"].fmeasure
            reward += rougel * 1
            similarity = similarity_score(extracted, gold)
            reward += similarity * 2
            # if step % args.logging_step == 0:
            print('---')
            print(f"rogue reward:{rougel}")
            print(f"similarity reward:{similarity}")
        
        rewards.append(reward)
        
    
    return rewards

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(script_args.dataset_name)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_process_match,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_new_tokens, do_sample=True, temperature=training_args.temperature
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    with open(os.path.join(training_args.output_dir, 'log.json'), 'w') as f:
        json.dump(trainer.state.log_history, f)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)