import numpy as np
from datasets import load_dataset, Dataset
import json
import argparse


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='generated/iter1')
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--num_gpu", type=int, default=8)
    return parser.parse_args()

def from_ranks(args):

    num_gpu = args.num_gpu
    pairs = args.pairs

    dataset = "HuggingFaceH4/ultrafeedback_binarized"
    data = load_dataset(dataset, split="train_prefs")

    scores = [0 for i in range(len(data))]

    for idx in range(num_gpu):
        locals = np.load(f"ranking/{args.output_dir}/{idx}.npy")
        locals = list(locals)
        for lidx, sc in enumerate(locals):
            scores[lidx*num_gpu+idx] = sc

    # print(scores[0])

    probs = []
    for idx, score in enumerate(scores):
        # prb[i][j] is the winning probability i over j
        prb = np.zeros((pairs+2,pairs+2)) # generated + original chosen + original rejected
        for i in range(pairs+2):
            for j in range(pairs+2):
                prb[i][j] = 1/(1+np.exp(score[j]-score[i]))
        prb = prb.tolist()
        probs.append(prb)

    # print(probs)
    # print(len(probs))
        
    print("Saving probabilities...")    

    f = open(f"generated/{args.output_dir}/probabilities.json", "w")
    json.dump(probs, f)
    f.close()

    df = data.to_pandas()
    for i in range(pairs):
        f = open(f"generated/{args.output_dir}/responses_{i}.json")
        responses = json.load(f)
        f.close()
        fmt = [
            [
                {"content": data[j]["prompt"], "role": "user"},
                {"content": responses[j], "role": "assistant"},
            ]
            for j in range(len(data))
        ]
        df[f"generate_{i}"] = fmt

    df["probability"] = probs
    df.to_parquet(f"generated/{args.output_dir}/all.parquet")

if __name__ == "__main__":
    args = parse_arguments()
    from_ranks(args)
    data = Dataset.from_parquet(f"generated/{args.output_dir}/all.parquet")
    data.push_to_hub(f"UCLAML/{args.output_dir}_generated")
