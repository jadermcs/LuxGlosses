import json
import pandas as pd

df = pd.read_csv("data/lod_english_word.csv", sep="\t")
df = df.dropna(subset="en_word")

test = open("test.jsonl", "w")

for mode in ["train", "test"]:
    filtered = df.lemma >= "J" if mode == "train" else df.lemma < "J"
    df_group = df[filtered].groupby("lemma").first().reset_index().sample(frac=1.)
    with open(f"{mode}.jsonl", "w") as fout:
        count = 0
        for idx, row in df_group.iterrows():
            word = row["en_word"]
            if not word[0].isalpha():
                continue
            sentence = row['sentence']
            lux_word = row["lemma"].lower()
            count += 1
            data = {"messages": [
                {"role": "system", "content": "You are a helpful assistant specialized in Luxembourgish language."},
                {"role": "user", "content": f"In the sentence '{sentence}' what word means '{word}'. Give me the word in it's base form (lemma)."},
                {"role": "assistant", "content": f"{lux_word}"}
                ]}
            fout.write(json.dumps(data)+"\n")
            if count > 1000:
                break
