import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

df = pd.read_csv("data/dewordnet.csv", sep="\t")


def get_definition(word):
    definitions = df[df.lemma == word]["definition"].values
    return definitions


def main():
    data = pd.read_csv("data/lod_german_word.csv", sep="\t")
    data = data.groupby("meaning").first()
    data["wn_definition"] = pd.NA
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        if not row["de_word"] or not row["de_definition"]:
            continue
        definitions = get_definition(row["de_word"]).tolist()
        if not definitions:
            continue
        sentences1 = model.encode(definitions)
        sentences2 = model.encode(row["de_definition"])

        similarities = model.similarity(sentences1, sentences2)
        argmax = similarities.argmax().item()
        data.loc[idx, "wn_definition"] = definitions[argmax]

    data.dropna().to_csv("data/german_definitions.csv", sep="\t", index=False)


if __name__ == "__main__":
    main()
