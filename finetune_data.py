import json
import pandas as pd

df = pd.read_csv("data/lod_english_word.csv", sep="\t")
df = df.dropna(subset="en_word")

test = open("test.jsonl", "w")

for mode in ["train", "test"]:
    filtered = df.lemma != ""
    df_group = df[filtered].groupby("lemma").first().reset_index().sample(frac=1.)
    with open(f"{mode}.jsonl", "w") as fout:
        count = 0
        for idx, row in df_group.iterrows():
            sentence = row['sentence']
            lux_word = row["lemma"]
            count += 1
            data = {"messages": [
                {"role": "system", "content": "You are a helpful assistant specialized in Luxembourgish language."},
                {"role": "user", "content": f"Give me a sentece in Luxembourgish using the word '{lux_word}'."},
                {"role": "assistant", "content": f"{sentence}"}
                ]}
            fout.write(json.dumps(data)+"\n")
