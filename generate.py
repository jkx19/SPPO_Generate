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

from alignment.model_utils import is_adapter_model
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

    # data = load_dataset(dataset, split="train_prefs")
        # print(data[0]["chosen"])
    # exit()
    # prompts = ["### Instruction: " + data[idx]["prompt"] + "\n\n### Response: " for idx in range(len(data))]
    f = open("fractions/full_prompts.json")
    prompts = json.load(f)
    f.close()

    # f = open(f"fractions/{data_frac}_of_8.json")
    # idx_prompts_all = json.load(f)
    # f.close()
    # prompts = [x[1] for x in idx_prompts_all]

    model = model_path
    # if is_adapter_model(model, None) is True:
    #     peft_config = PeftConfig.from_pretrained(model_path, revision=None)
    #     model_kwargs = dict(
    #         revision=None,
    #         torch_dtype=torch.bfloat16,
    #         device_map={"": accelerator.process_index},
    #     )
    #     base_model = AutoModelForCausalLM.from_pretrained(
    #         peft_config.base_model_name_or_path,
    #         **model_kwargs,
    #     )
    #     model = PeftModel.from_pretrained(
    #         base_model,
    #         model_path,
    #         revision=None
    #     )
    #     tokenizer = AutoTokenizer.from_pretrained(model_path)   
    #     tokenizer.pad_token = tokenizer.eos_token
    
    # else:
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,    
    #     device_map={"": accelerator.process_index},
    #     torch_dtype=torch.bfloat16,
    # )
    tokenizer = AutoTokenizer.from_pretrained(model_path)   
    tokenizer.pad_token = tokenizer.eos_token
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.world_size,
    )
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=512)


    pairs = args.pairs   

    for p in range(pairs):
        response = llm.generate(prompts, sampling_params)
        output = list(map(lambda x: x.outputs[0].text, response))
        f = open(f"{args.output_dir}/responses_{p}.json", "w")
        json.dump(output, f)
        f.close()

        # print("-------------------------------------------------------")
        # print(output[0])

    # results = [[] for i in range(pairs)]
    
    # batch_size = args.batch_size
    # prompt_batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)] 
    # tokenizer.padding_side="left"

    # for prompts_batch in tqdm(prompt_batches):
    #     prompts_tokenized = tokenizer(
    #             prompts_batch, 
    #             return_tensors="pt", 
    #             padding='longest', 
    #             truncation=True,
    #             max_length=1024, 
    #             pad_to_multiple_of=8,
    #             add_special_tokens=False).to("cuda") 
    #     # set max_new_tokens smaller for faster inference

    #     for p in range(pairs):
    #         # outputs_tokenized=model.generate(**prompts_tokenized, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id, do_sample=True)
    #         outputs_tokenized = llm.generate()

    #         # remove prompt from gen. tokens
    #         outputs_tokenized=[ tok_out[len(tok_in):] 
    #             for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 
    #         # decode gen. tokens 
    #         outputs=tokenizer.batch_decode(outputs_tokenized)
    #         results[p].extend(outputs)
        # exit()
        # results.extend(outputs)


    # if accelerator.is_local_main_process:
    #     timediff=time.time()-start
    #     print(f"time elapsed: {timediff}")

    #     # collecting data
    #     for idx in range(len(chosen_all)):
    #         d = {"chosen": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": chosen_all[idx]}], "generated": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": results[idx]}]}
    #         if args.split == 'test':
    #             filename = f"{args.output_dir}/loser_{data_frac}_test.jsonl"
    #         else:
    #             filename = f"{args.output_dir}/loser_{data_frac}.jsonl"
    #         with open(filename, 'a') as f:
    #             json.dump(d, f)
    #             f.write('\n')


if __name__ == "__main__":
    main()