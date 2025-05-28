from evaluate import load
import pandas as pd

data = pd.read_csv("data/Evaluation LuxGlosses.csv")

data = data[data['annotator']=='Fred']

ter = load("ter")
bleu = load("bleu")
chrf = load("chrf")

results = ter.compute(predictions=data['lux_definition'].tolist(),
                      references=data['necessary_edits'].tolist())

bleu_results = bleu.compute(predictions=data['lux_definition'].tolist(),
                           references=data['necessary_edits'].tolist())

chrf_results = chrf.compute(predictions=data['lux_definition'].tolist(),
                            references=data['necessary_edits'].tolist(),
                            char_order=6,
                            word_order=2,
                            beta=2)

print("BLEU:", bleu_results)
print("chrf++:", chrf_results)
print("TER:", results)



