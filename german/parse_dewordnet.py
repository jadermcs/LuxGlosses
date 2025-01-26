import xml.etree.ElementTree as ET
import pandas as pd


tree = ET.parse('data/deWordNet.xml')
words = dict()
root = tree.getroot()
lemmas = []
definitions = []

for entry in root:
    for lex in entry.findall("./LexicalEntry"):
        lex_name = lex.find("./Lemma").attrib["writtenForm"]
        for lex_id in lex.findall("./Sense"):
            lex_id = lex_id.attrib["synset"]
            lemmas.append({
                "lemma": lex_name,
                "id": lex_id,
                })
    for synset in entry.findall("./Synset"):
        lex_id = synset.attrib["id"]
        lex_def = synset.find("./Definition")
        if lex_def is not None:
            lex_def = lex_def.text
            definitions.append({
                "definition": lex_def,
                "id": lex_id,
                })

df = pd.DataFrame(lemmas).set_index("id")
lex = pd.DataFrame(definitions).set_index("id")
df = df.join(lex).dropna().reset_index()
df.to_csv("data/dewordnet.csv", sep="\t", index=False)
