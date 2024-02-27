import numpy as np
from datasets import load_dataset, Dataset
import json

def from_ranks():

    num_gpu=8
    pairs = 3

    dataset = "HuggingFaceH4/ultrafeedback_binarized"
    data = load_dataset(dataset, split="train_prefs")

    scores = [0 for i in range(len(data))]

    for idx in range(num_gpu):
        locals = np.load(f"ranking/mistral/{idx}.npy")
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

    f = open("generated/mistral/probabilities.json", "w")
    json.dump(probs, f)
    f.close()

    df = data.to_pandas()
    for i in range(pairs):
        f = open(f"generated/mistral/responses_{i}.json")
        responses = json.load(f)
        f.close()
        df[f"generate_{i}"] = responses

    df["probability"] = probs
    df.to_parquet("generated/mistral/all.parquet")


data = Dataset.from_parquet("generated/mistral/mistral.parquet")
data.push_to_hub("jikaixuan/mistral_generated")
