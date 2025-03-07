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
        data = [
            {"role": "system", "content": "You are a helpful assistant specialized in Luxembourgish language."},
            {"role": "user", "content": f"In the sentence '{sentence}' what is the definition of '{word}'? Give the definition in Luxembourgish. After reasoning give the answer in the last line."},
            ]
        print(data)
        completion = client.chat.completions.create(
            model=model_name,
            messages=data
        )

        response = completion.choices[0].message.content.split("\n")[-1]
        print(response)
        pred.append(response)
        if row["definition2"]:
            ref.append([row["definition1"], row["definition2"]])
        else:
            ref.append([row["definition1"]])
        break

    results = bleu.compute(predictions=pred, references=ref)
    print(results)

