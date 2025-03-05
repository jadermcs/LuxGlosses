from openai import OpenAI
import spellux
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
    {"role": "system", "content": "You are a helpful assistant that translates glosses from English to Luxembourgish. The definition of a word should not contain the word itself."}
]

for j in range(len(few_shot_samples)):
    model_inputs.append({"role": "user", "content": f"English word: {few_shot_samples.loc[j, 'word_en']}\nLuxembourgish word translation: {few_shot_samples.loc[j, 'word_lb']}\nLuxembourgish word in example sentence: {few_shot_samples.loc[j, 'usage']}\nEnglish word definition: {few_shot_samples.loc[j, 'definition_en']}"})
    model_inputs.append({"role": "assistant", "content": f"Luxembourgish translation of definition: {few_shot_samples.loc[j, 'definition_lb']}"})

# Load data
data = pd.read_csv('data/english_definitions.csv', delimiter='\t')
data = data[data['confidence']>=0.8]

for i, row in data[:50].iterrows():

    messages = model_inputs + [{"role": "user", "content": f"English word: {row['en_word']}\nLuxembourgish word translation: {row['lemma']}\nLuxembourgish word in example sentence: {row['sentence']}\nEnglish word definition: {row['wn_definition']}"}]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages).choices[0].message.content

    lux_definition = response.replace('Luxembourgish translation of definition: ', '')

    corrected = spellux.normalize_text(lux_definition, stats=False)
    if corrected != lux_definition:
        messages += [{"role": "user", "content": f"The spell checker said it is not written correcly, can you verify your answer."}]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages).choices[0].message.content
        lux_definition = response.splitlines().replace('Luxembourgish translation of definition: ', '')

    data.loc[i, 'lux_definition'] = lux_definition

    print('English word: ', row['en_word'])
    print('Luxembourgish word: ', row['lemma'])
    print('English definition: ', row['wn_definition'])
    print('Luxembourgish translation: ', lux_definition)

data.dropna().to_csv("data/lod_lux_definitions_fewshot.csv", sep="\t", index=False)
