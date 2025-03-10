import os
import pandas as pd
import evaluate
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

glosses = pd.read_csv("glosses.tsv", sep="\t")

for model_name in ["gpt-4o-mini", "ft:gpt-4o-mini-2024-07-18:list:lod-translate:B7n1076x"]:
    ref = []
    pred = []
    count = 0
    for idx, row in tqdm(glosses.iterrows(), total=glosses.shape[0]):
        sentence = row["usage"]
        word = row["word"]
        data = [
            {"role": "system", "content": "You are a helpful assistant specialized in Luxembourgish language."},
            {"role": "user", "content": f"In the sentence '{sentence}' what is the definition of '{word}'? Give the definition in Luxembourgish. After reasoning give the answer in the last line."},
            ]
        completion = client.chat.completions.create(
            model=model_name,
            messages=data
        )

        response = completion.choices[0].message.content.split("\n")[-1].split(":")[-1]
        pred.append(response)
        defs = [x for x in row[["definition1", "definition2"]].dropna().tolist() if x]
        ref.append(defs)
        count += 1

    with open(f"{model_name}.txt", "w") as fout:
        for line in pred:
            fout.write(line+"\n")

