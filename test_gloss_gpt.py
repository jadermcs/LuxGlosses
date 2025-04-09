import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

glosses = pd.read_csv("data/english_definitions.csv", sep="\t")
glosses = glosses[glosses.confidence > 0.6].groupby("wn_definition").first().reset_index()
print(glosses)

for model_name in ["ft:gpt-4o-mini-2024-07-18:list:lets-phrase:B9V61v1e"]:
    pred = []
    count = 0
    for idx, row in tqdm(glosses.iterrows(), total=glosses.shape[0]):
        sentence = row["wn_definition"]
        data = [
            {"role": "system", "content": "You are a helpful assistant specialized in Luxembourgish language."},
            {"role": "user", "content": f"Translate the sentence '{sentence}' to Luxembourgish."},
            ]
        completion = client.chat.completions.create(
            model=model_name,
            messages=data
        )

        response = completion.choices[0].message.content.split("\n")[-1].split(":")[-1]
        pred.append(response)
        count += 1

    glosses["gpt_fine_tune"] = pred
    glosses.to_csv(f"{model_name}_definition.csv", sep="\t", index=False)
