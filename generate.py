# reference: https://medium.com/@geronimo7/llms-multi-gpu-inference-with-accelerate-5a8333e4c5db

from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams

import argparse
import torch, time, json, os
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

import warnings
warnings.filterwarnings("ignore")

kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
accelerator = Accelerator(kwargs_handlers=[kwargs])

from peft import PeftConfig, PeftModel

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0')
    # parser.add_argument('--data_frac', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='generated/iter1')
    parser.add_argument('--maxlen', type=int, default=1024)
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--world_size', type=int, default=4)
    return parser.parse_args()

# def prepare_prompts(prompts, tokenizer, batch_size=4):
#     """Prepare prompts for tokenization."""
#     # prompts = [x[1] for x in idx_prompts]
#     # idcs = [x[0] for x in idx_prompts]
#     batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)] 
#     # batch_ids = [idcs[i:i + batch_size] for i in range(0, len(prompts), batch_size)] 
#     batches_tok=[]
#     tokenizer.padding_side="left"     
#     for prompt_batch in batches:
#         batches_tok.append(
#             tokenizer(
#                 prompt_batch, 
#                 return_tensors="pt", 
#                 padding='longest', 
#                 truncation=False, 
#                 pad_to_multiple_of=8,
#                 add_special_tokens=False).to("cuda") 
#             )
#     tokenizer.padding_side="right"
#     return batches_tok

def main():
    args = parse_arguments()
    model_path = args.model
    # data_frac = args.data_frac
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = "HuggingFaceH4/ultrafeedback_binarized"
    data = load_dataset(dataset, split="train_prefs")
    prompts = ["### Instruction: " + data[idx]["prompt"] + "\n\n### Response: " for idx in range(len(data))]

    tokenizer = AutoTokenizer.from_pretrained(model_path)   
    tokenizer.pad_token = tokenizer.eos_token
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.world_size,
    )
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.maxlen, seed=args.seed)


    pairs = args.pairs   

    os.makedirs(args.output_dir, exist_ok=True)

    for p in range(pairs):
        response = llm.generate(prompts, sampling_params)
        output = list(map(lambda x: x.outputs[0].text, response))
        f = open(f"{args.output_dir}/responses_{p}.json", "w")
        json.dump(output, f)
        f.close()

        


if __name__ == "__main__":
    main()