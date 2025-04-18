from openai import OpenAI
import pandas as pd
import os

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Prepare few-shot samples
few_shot_samples = pd.DataFrame({
    'word_lb': ['Aarm', 'bludden', 'Concours', 'Duerf', 'Haus'],
    'definition_en': [
        "A limb of the body extending from the shoulder to the hand",
        "To lose blood from the body, typically through a wound or injury",
        "An event in which individuals or groups compete for a prize or recognition",
        "A small settlement, typically in a rural area, smaller than a town",
        "A building for human habitation, typically where a family lives"
        ],
    'definition_lb': [
        "E Glidd vum Kierper, dat vun der Schëller bis bei d'Hand geet",
        "Blutt aus dem Kierper verléieren, meeschtens duerch eng Wonn oder Verletzung",
        "En Evenement, wou eenzel Leit oder Gruppen ëm e Präis oder Unerkennung konkurréieren",
        "Eng kleng Uertschaft, normalerweis um Land, méi kleng wéi eng Stad",
        "E Gebai an deem Mënsche wunnen, normalerweis wou eng Famill lieft"
        ],
    })

model_inputs = [
    {"role": "system", "content": "You are a helpful assistant that translates glosses from English to Luxembourgish. Don't use the original word in Luxembourgish for the definition."}
]

for j in range(2):
   model_inputs.append({"role": "user", "content": f"Luxembourgish word: {few_shot_samples.loc[j, 'word_lb']}\nEnglish word definition: {few_shot_samples.loc[j, 'definition_en']}"})
   model_inputs.append({"role": "assistant", "content": f"Luxembourgish translation of definition: {few_shot_samples.loc[j, 'definition_lb']}"})

# Load data
data = pd.read_csv('data/english_definitions.csv', delimiter='\t')
data = data[data['confidence'] > 0.6]
data = data[~data['code'].str.endswith("_EGS")]
print(data.shape)
exit()

try:
    for i, row in data.iterrows():
        print("Processing example", i)
        messages = model_inputs + [{"role": "user", "content": f"Luxembourgish word: {row['lemma']}\nEnglish word definition: {row['wn_definition']}"}]

        response = client.chat.completions.create(
            model="gpt-4.5-preview",
            messages=messages).choices[0].message.content

        lux_definition = response.replace('Luxembourgish translation of definition: ', '')

        data.loc[i, 'lux_definition'] = lux_definition

        print('English word: ', row['en_word'])
        print('Luxembourgish word: ', row['lemma'])
        print('English definition: ', row['wn_definition'])
        print('Luxembourgish translation: ', lux_definition)
        print('Confidence: ', data.loc[i, 'confidence'])
except KeyboardInterrupt:
    print("interrupted")

data.dropna().to_csv("data/lod_lux_definitions_fewshot_4.5.csv", sep="\t", index=False)
