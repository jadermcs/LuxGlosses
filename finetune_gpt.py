import json
import random
from datasets import load_dataset

letz = load_dataset("fredxlpy/LETZ", "LETZ-SYN")

words = letz["train"]["label"]

NUM = 4
count = 0

with open("train.jsonl", "w") as fout:
    for example in letz["train"]:
        count += 1
        choices = random.choices(words, k=NUM)
        i = random.randint(0, NUM-1)
        word = example["label"]
        choices[i] = word
        options = "\n".join([f"{x}) {choices[idx]}" for idx, x in enumerate("ABCD")])
        data = {"messages": [
            {"role": "system", "content": "You are a helpful assistant specialized in Luxembourgish meaning."},
            {"role": "user", "content": f"What synonym word is contained in '{example['text']}': \n{options}"},
            {"role": "assistant", "content": f"{'ABCD'[i]}) {word}"}
            ]}
        fout.write(json.dumps(data)+"\n")
        if count >= 100:
            break
