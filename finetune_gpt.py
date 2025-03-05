import json
import pandas as pd

df = pd.read_csv("data/lod_english_word.csv", sep="\t").sample(frac=1.)

test = open("test.jsonl", "w")

for mode in ["train", "test"]:
    filtered = df.lemma <= "J"
    if mode == "test":
        filtered != filtered
    with open(f"{mode}.jsonl", "w") as fout:
        count = 0
        for idx, row in df[df.lemma <= "J"].groupby("en_word").first().reset_index().iterrows():
            count += 1
            word = row["en_word"]
            lux_word = row["lemma"]
            data = {"messages": [
                {"role": "system", "content": "You are a helpful assistant specialized in Luxembourgish language."},
                {"role": "user", "content": f"In the sentence '{row['sentence']}' what is the meaning of '{lux_word}', answer with maximum 5 words."},
                {"role": "assistant", "content": f"{word}"}
                ]}
            fout.write(json.dumps(data)+"\n")
            if count >= 1000:
                break
