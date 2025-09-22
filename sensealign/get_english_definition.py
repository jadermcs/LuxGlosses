import pandas as pd
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def get_definition(word, pos):
    if pos in ["NP", "SUBST", "n", "INTERJ", "CONJ"]:
        pos = wn.NOUN
    elif pos in ["ADJ", "a"]:
        pos = wn.ADJ
    elif pos in ["VRB", "v"]:
        pos = wn.VERB
    elif pos in ["ADV"]:
        pos = wn.ADV
    else:
        print(f"Not found {pos} for {word}")
        return []
    synsets = wn.synsets(word, pos=pos)
    definitions = [x.definition() for x in synsets]
    return definitions


def main():
    data = pd.read_csv("data/lod_english_word.csv", sep="\t")
    data["code"] = data["meaning"]
    data = data.groupby("meaning").first()
    data["wn_definition"] = pd.NA
    data["confidence"] = pd.NA
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        if not row["en_word"] or not row["en_definition"] or row["code"].endswith("_EGS"):
            continue
        definitions = get_definition(row["en_word"].removeprefix("to "),
                                     row["pos"])
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
