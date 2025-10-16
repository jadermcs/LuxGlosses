import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import wn as WN
import fasttext
from huggingface_hub import hf_hub_download
try:
    from sensealign.french_wordnet import WordNet
except ImportError:
    from french_wordnet import WordNet

# Load WordNets
# WN.download('odenet:1.4') # German
from nltk.corpus import wordnet as wn_en
wn_de = WN.Wordnet("odenet:1.4")
wn_fr = WordNet.from_xml("data/wonef.xml")

# Load sentence embedding models
model_en = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
model_de = SentenceTransformer("jinaai/jina-embeddings-v2-base-de", trust_remote_code=True)

# Load LID model
model_path = hf_hub_download(repo_id="laurievb/OpenLID", filename="model.bin")
lid_model = fasttext.load_model(model_path)


def get_definition(word, pos, lang):
    if pos in ["NP", "SUBST", "n", "INTERJ", "CONJ"]:
        pos = wn_en.NOUN if lang == "en" else "n"
    elif pos in ["ADJ", "a"]:
        pos = wn_en.ADJ if lang == "en" else "a"
    elif pos in ["VRB", "v"]:
        pos = wn_en.VERB if lang == "en" else "v"
    elif pos in ["ADV"]:
        pos = wn_en.ADV if lang == "en" else "adv"
    else:
        print(f"Not found {pos} for {word}")
        return []
    wn = wn_en if lang == "en" else wn_fr if lang == "fr" else wn_de
    synsets = wn.synsets(word, pos=pos)
    definitions = [x.definition() for x in synsets]
    return definitions

def find_max_similarity(sentences1, sentences2, model):
    sentences1 = model.encode(sentences1)
    sentences2 = model.encode(sentences2)
    similarities = model.similarity(sentences1, sentences2)
    argmax = similarities.argmax().item()
    valuemax = similarities.max().item()
    return argmax, valuemax


def main():
    data = pd.read_csv("data/lod_multilingual_words.csv", sep="\t")
    data["code"] = data["meaning"]
    data = data.groupby("meaning").first()
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        if row["code"].endswith("_EGS"):
            continue
        
        # Get definitions from WordNets
        if row["en_word"]:
            if not row["en_definition"]:
                row["en_definition"] = row["en_word"]
            en_definitions = get_definition(row["en_word"].removeprefix("to "), row["pos"], "en")
        if row["de_word"]:
            if not row["de_definition"]:
                row["de_definition"] = row["de_word"]
            de_definitions = get_definition(row["de_word"], row["pos"], "de")
            de_definitions = [d for d in de_definitions if d!=None]
            de_definitions = [d for d in de_definitions if ((len(d.split())>=3) and (row["de_word"] not in d))]
        if row["fr_word"]:
            if not row["fr_definition"]:
                row["fr_definition"] = row["fr_word"]
            fr_definitions = get_definition(row["fr_word"], row["pos"], "fr")
            fr_definitions = [d for d in fr_definitions if d!=None]
            fr_definitions = [d for d in fr_definitions if lid_model.predict([d.replace("\n", " ")])[0][0][0]== "__label__eng_Latn"]

        if not en_definitions and not de_definitions and not fr_definitions:
            continue

        # EN
        if en_definitions and row["en_word"] and row["en_definition"]:
            argmax, valuemax = find_max_similarity(
                en_definitions, [row["en_definition"]], model_en)
            if valuemax >= 0.6:  # threshold
                data.loc[idx, "wn_definition_en"] = en_definitions[argmax]
                data.loc[idx, "confidence_en"] = valuemax

        # DE
        if de_definitions and row["de_word"] and row["de_definition"]:
            argmax, valuemax = find_max_similarity(
                de_definitions, [row["de_definition"]], model_de)
            if valuemax >= 0.6:  # threshold
                data.loc[idx, "wn_definition_de"] = de_definitions[argmax]
                data.loc[idx, "confidence_de"] = valuemax

        # FR (definitions are always in English)
        if fr_definitions and row["en_word"] and row["en_definition"]:
            argmax, valuemax = find_max_similarity(
                fr_definitions, [row["en_definition"]], model_en)
            if valuemax >= 0.6:  # threshold
                data.loc[idx, "wn_definition_fr"] = fr_definitions[argmax]
                data.loc[idx, "confidence_fr"] = valuemax

    # Make final decision on which definitions to keep (First EN, then FR, then DE)
    for i, row in data.iterrows():
        if not pd.isna(row['wn_definition_en']):
            data.loc[i, 'wn_definition'] = row['wn_definition_en']
            data.loc[i, 'definition_language'] = 'en'
            data.loc[i, 'confidence'] = row['confidence_en']
            continue
        if not pd.isna(row['wn_definition_fr']):
            data.loc[i, 'wn_definition'] = row['wn_definition_fr']
            data.loc[i, 'definition_language'] = 'fr'
            data.loc[i, 'confidence'] = row['confidence_fr']
        if not pd.isna(row['wn_definition_de']):
            data.loc[i, 'wn_definition'] = row['wn_definition_de']
            data.loc[i, 'definition_language'] = 'de'
            data.loc[i, 'confidence'] = row['confidence_de']
            continue

    data.dropna(subset=['wn_definition']).to_csv("data/multilingual_definitions.csv", sep="\t", index=False)

if __name__ == "__main__":
    main()