from openai import OpenAI
import pandas as pd
import os

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Load data
data = pd.read_csv('data/english_definitions.csv', delimiter='\t')
data = data[data['confidence']>0.6]

# Ignore already translated definitions
output_file = "data/lod_lux_definitions_zeroshot_4.5.csv"
if os.path.exists(output_file):
    data_existing = pd.read_csv(output_file, delimiter="\t")
    data_existing = data_existing[data_existing['lux_definition'].notna()]
    data = data[~data['code'].isin(data_existing['code'])]

for i, row in data.iterrows():

    messages = [{"role": "user", "content": f"Translate the English definition to Luxembourgish. Do not use the original word in Luxembourgish for the definition. Stay as close as possible to the English definition, but modify it if the Luxembourgish translation would otherwise sound unnatural. Only return the translation, nothing else.\nLuxembourgish word: {row['lemma']}\nEnglish word definition: {row['wn_definition']}"}]

    response = client.chat.completions.create(
        model="gpt-4.5-preview",
        messages=messages,
        temperature=0).choices[0].message.content
    
    lux_definition = response
    
    data.loc[i, 'lux_definition'] = lux_definition

    print('English word: ', row['en_word'])
    print('Luxembourgish word: ', row['lemma'])
    print('English definition: ', row['wn_definition'])
    print('Luxembourgish translation: ', lux_definition)

# Add the new translations to the existing file (if it exists)
if os.path.exists(output_file):
    pd.concat([data_existing, data.dropna()], ignore_index=True).to_csv(output_file, sep="\t", index=False)
else:
    data.dropna().to_csv(output_file, sep="\t", index=False)
