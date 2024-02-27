from datasets import load_dataset
import json
from tqdm import tqdm

dataset = "HuggingFaceH4/ultrafeedback_binarized"

data = load_dataset(dataset, split="train_prefs")
    # print(data[0]["chosen"])
# exit()
prompts_all = ["### Instruction: " + data[idx]["prompt"] + "\n\n### Response: " for idx in range(len(data))]
idxed_prompts = [(i, p) for i, p in enumerate(prompts_all)]

frac = 8

frac_prompts = []
for res in tqdm(range(frac)):
    frac_res = [x for x in idxed_prompts if x[0]%frac==res]
    f = open(f"fractions/{res}_of_{frac}.json", "w")
    json.dump(frac_res, f)
    f.close()



# prompts_old = [data[idx]["prompt"] for idx in range(len(data))]
# chosen_all = [data[idx]["chosen"][1]["content"] for idx in range(len(data))]