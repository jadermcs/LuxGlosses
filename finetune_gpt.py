import os
import json
import random
from openai import OpenAI
from datasets import load_dataset

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

letz = load_dataset("fredxlpy/LETZ", "LETZ-SYN")

words = letz["train"]["label"]

NUM = 4

with open("train.json", "w") as fout:
    for example in letz["train"]:
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

# response = client.fine_tuning.jobs.create(
#     training_file="file-abc123",
#     model="gpt-4o-mini-2024-07-18"
# )


