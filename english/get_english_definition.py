import pandas as pd
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def get_definition(word):
    synsets = wn.synsets(word)
    definitions = [x.definition() for x in synsets]
    return definitions


def main():
    data = pd.read_csv("data/lod_english_word.csv", sep="\t")
    data["code"] = data["meaning"]
    data = data.groupby("meaning").first()
    data["wn_definition"] = pd.NA
    data["confidence"] = pd.NA
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        if not row["en_word"] or not row["en_definition"]:
            continue
        definitions = get_definition(row["en_word"].removeprefix("to "))
        if not definitions:
            continue
        sentences1 = model.encode(definitions)
        sentences2 = model.encode(row["en_definition"])

        similarities = model.similarity(sentences1, sentences2)
        argmax = similarities.argmax().item()
        valuemax = similarities.max().item()
        data.loc[idx, "wn_definition"] = definitions[argmax]
        data.loc[idx, "confidence"] = valuemax

    data.dropna().to_csv("data/english_definitions.csv", sep="\t", index=False)


if __name__ == "__main__":
    main()
