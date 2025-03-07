import os
import pandas as pd
import evaluate
from openai import OpenAI

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

glosses = pd.read_csv("glosses.tsv", sep="\t")

bleu = evaluate.load("bleu")

for model_name in ["gpt-4o-mini", "ft:gpt-4o-mini-2024-07-18:list:lod-translate:B7n1076x"]:
    ref = []
    pred = []
    for idx, row in glosses.iterrows():
        sentence = row["usage"]
        word = row["word"]
        data = {"messages": [
            {"role": "system", "content": "You are a helpful assistant specialized in Luxembourgish language."},
            {"role": "user", "content": f"In the sentence '{sentence}' what is the dictionary meaning of '{word}'. After reasoning give the answer in the last line."},
            ]}
        completion = client.chat.completions.create(
            model=model_name,
            messages=data
        )

        response = completion.choices[0].message.content.split("\n")[-1]
        pred.append(response)
        if row["usage2"]:
            ref.append([row["usage1"], row["usage2"]])
        else:
            ref.append([row["usage1"]])

    print(bleu.compute(pred, ref))
