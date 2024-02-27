from datasets import load_dataset, Dataset
import json
from tqdm import tqdm

import pandas as pd
import argparse
import llm_blender
import os
import numpy as np
from copy import deepcopy

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='generated/iter1')
    # parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument("--multi_thread", action="store_true")
    parser.add_argument("--numgpu", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0) # local rank
    parser.add_argument("--pairs", type=int, default=1)
    # parser.add_argument('--world_size', type=int, default=4)
    return parser.parse_args()


def ranking(prompts, candidates, gpu, outdir):
    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM", device=f"cuda:{gpu}") # load PairRM
    ranks = blender.rank(prompts, candidates, return_scores=True, batch_size=1)
    np.save(f"ranking/{outdir}/{gpu}.npy", ranks)


def main(args):
    dataset = "HuggingFaceH4/ultrafeedback_binarized"
    data = load_dataset(dataset, split="train_prefs")
    # df:pd.DataFrame = data.to_pandas()[["prompt", "chosen", "rejected", "score_chosen", "score_rejected"]]

    prompts_all = ["### Instruction: " + data[idx]["prompt"] + "\n\n### Response: " for idx in range(len(data))]
    chosen_all = [data[idx]["chosen"][1]["content"] for idx in range(len(data))]
    reject_all = [data[idx]["rejected"][1]["content"] for idx in range(len(data))]
    # idxed_prompts = [(i, p) for i, p in enumerate(prompts_all)]

    pairs = args.pairs

    all_generated = []

    for i in range(pairs):
        f = open(f"generated/{args.output_dir}/responses_{i}.json")
        gen = json.load(f)
        all_generated.append(gen)
        f.close()

    # blender = llm_blender.Blender()
    # blender.loadranker("llm-blender/PairRM") # load PairRM

    candidates_texts = list(zip(chosen_all, reject_all, *all_generated))

    prompts_all = prompts_all[:100]
    candidates_texts = candidates_texts[:100]

    if args.multi_thread:
        import threading
        threadlist: list[threading.Thread] = []
        for gpu in range(args.numgpu):
            local_prompts = [p for i, p in enumerate(prompts_all) if i%args.numgpu == gpu]
            local_candidates = [c for i, c in enumerate(candidates_texts) if i%args.numgpu == gpu]
            t = threading.Thread(
                target=ranking, 
                args=(deepcopy(local_prompts), deepcopy(local_candidates), gpu, args.output_dir)
            )
            threadlist.append(t)

        for t in threadlist:
            t.start()
        for t in threadlist:
            t.join()
    
    else:
        blender = llm_blender.Blender()
        blender.loadranker("llm-blender/PairRM") # load PairRM
        gpu = args.gpu
        prompts_all = [p for i, p in enumerate(prompts_all) if i%args.numgpu == gpu]
        candidates_texts = [c for i, c in enumerate(candidates_texts) if i%args.numgpu == gpu]
        ranks = blender.rank(prompts_all, candidates_texts, return_scores=True, batch_size=1)
        os.makedirs(f"ranking/{args.output_dir}", exist_ok=True)
        np.save(f"ranking/{args.output_dir}/{gpu}.npy", ranks)


# print(ranks)

# import numpy as np
# for idx in range(len(ranks)):
#     rank = np.array(ranks[idx])
#     win = np.argmax(rank)
#     lose = np.argmin(rank)
#     df["chosen"][idx][1]["content"] = candidates_texts[idx][win]
#     df["rejected"][idx][1]["content"] = candidates_texts[idx][lose]
#     p_win = 1/(1 + np.exp(-rank[win]+rank[lose]))
#     df["score_chosen"][idx] = p_win
#     df["score_rejected"][idx] = 1 - p_win
#     # print(win, lose)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
