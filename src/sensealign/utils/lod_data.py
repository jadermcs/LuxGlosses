import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

np.random.seed(42)

tree = ET.parse('data/new_lod-art.xml')
words = set()
root = tree.getroot()
data = []

for entry in root:
    lemma = entry.find("./lemma").text
    pos = entry.find("./microStructure/partOfSpeech")
    if pos is not None:
        pos = pos.text
    for meaning in entry.findall("./microStructure/grammaticalUnit/meaning"):
        # EN
        word = meaning.find("./targetLanguage[@lang='en']/translation")
        clar = meaning.find("./targetLanguage[@lang='en']/semanticClarifier")
        if word is not None:
            word = word.text
        if clar is not None:
            clar = clar.text

        # DE
        de_word = meaning.find("./targetLanguage[@lang='de']/translation")
        de_clar = meaning.find("./targetLanguage[@lang='de']/semanticClarifier")
        if de_word is not None:
            de_word = de_word.text
        if de_clar is not None:
            de_clar = de_clar.text

        # FR
        fr_word = meaning.find("./targetLanguage[@lang='fr']/translation")
        fr_clar = meaning.find("./targetLanguage[@lang='fr']/semanticClarifier")
        if fr_word is not None:
            fr_word = fr_word.text
        if fr_clar is not None:
            fr_clar = fr_clar.text

        for e in meaning.findall("./examples/example/text"):
            meaning_txt = meaning.attrib["id"]
            string = ""
            for i in e:
                text = i.text
                # if EGS it is used colloquially, many times as a metaphor
                if text == "EGS":
                    meaning_txt += "_EGS"
                else:
                    string += text
                    string += "" if text.endswith("'") else " "
            words.add(lemma)
            data.append({
                "lemma": lemma,
                "pos": pos,
                "meaning": meaning_txt,
                "en_word": word,
                "en_definition": clar,
                "de_word": de_word,
                "de_definition": de_clar,
                "fr_word": fr_word,
                "fr_definition": fr_clar,
                "sentence": string})

df = pd.DataFrame(data)
df.to_csv("data/lod_multilingual_words.csv", sep="\t", index=False)
