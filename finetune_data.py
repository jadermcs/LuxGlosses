import json
import pandas as pd

df = pd.read_csv("data/lod_english_word.csv", sep="\t")
df = df.dropna(subset="en_word")
df = df.sample(frac=1.)

test = open("test.jsonl", "w")

for mode in ["train", "test"]:
    filtered = df.lemma <= "J"
    if mode == "test":
        filtered != filtered
    df_group = df[filtered].groupby("en_word").first().reset_index()
    with open(f"{mode}.jsonl", "w") as fout:
        count = 0
        for idx, row in df_group.iterrows():
            count += 1
            word = row["en_word"]
            lux_word = row["lemma"]
            data = {"messages": [
                {"role": "system", "content": "You are a helpful assistant specialized in Luxembourgish language."},
                {"role": "user", "content": f"In the sentence '{row['sentence']}' what word means '{word}'. Give me the word in it's base form (lemma)."},
                {"role": "assistant", "content": f"{lux_word}"}
                ]}
            fout.write(json.dumps(data)+"\n")
            if count >= 1000:
                break
