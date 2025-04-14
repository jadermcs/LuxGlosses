from openai import OpenAI
import pandas as pd
import os

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Prepare few-shot samples
few_shot_samples = pd.DataFrame({
    'word_lb': ['Aarm', 'bludden', 'Concours', 'Duerf', 'Haus'],
    'word_en': ['arm', 'to bleed', 'competition', 'village', 'house'],
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
    'usage': [
        "kuck, du hues en Himmelsdéierchen um Aarm sëtzen!",
        "no där äiseger Wanternuecht war de Séi zougefruer",
        "den Architekt krut am Concours den éischte Präis",
        "ech sinn an en anert Duerf geplënnert",
        "mir hunn nach vill Aarbecht ronderëm eist neit Haus"
        ]
    })

model_inputs = [
    {"role": "system", "content": "Du bass en hëllefsbereeten Assistent, deen Definitiounen aus dem Engleschen op Lëtzebuergesch iwwersetzt."}
]

for j in range(len(few_shot_samples)):
    model_inputs.append({"role": "user", "content": f"Englescht Wuert: {few_shot_samples.loc[j, 'word_en']}\nLëtzebuergesch Iwwersetzung vum Wuert: {few_shot_samples.loc[j, 'word_lb']}\nLëtzebuergescht Wuert an engem Beispillsaz: {few_shot_samples.loc[j, 'usage']}\nEnglesch Definitioun vum Wuert: {few_shot_samples.loc[j, 'definition_en']}"})
    model_inputs.append({"role": "assistant", "content": f"Lëtzebuergesch Iwwersetzung vun der Definitioun: {few_shot_samples.loc[j, 'definition_lb']}"})

# Load data
data = pd.read_csv('data/english_definitions.csv', delimiter='\t')
data = data[data['confidence']>=0.8]

for i, row in data[:50].iterrows():

    messages = model_inputs + [{"role": "user", "content": f"Englescht Wuert: {row['en_word']}\nLëtzebuergesch Iwwersetzung vum Wuert: {row['lemma']}\nLëtzebuergescht Wuert an engem Beispillsaz: {row['sentence']}\nEnglesch Definitioun vum Wuert: {row['wn_definition']}"}]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages).choices[0].message.content
    
    lux_definition = response.replace('Lëtzebuergesch Iwwersetzung vun der Definitioun: ', '')
    
    data.loc[i, 'lux_definition'] = lux_definition

    print('English word: ', row['en_word'])
    print('Luxembourgish word: ', row['lemma'])
    print('English definition: ', row['wn_definition'])
    print('Luxembourgish translation: ', lux_definition)

data.dropna().to_csv("data/lod_lux_definitions_fewshot_v2.csv", sep="\t", index=False)



