from openai import OpenAI
import pandas as pd
import os
from google import genai
from google.genai import types

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
# client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY_Trux"))


# Language mapping
lang_map = {
    'fr': 'English',
    'de': 'German',
    'en': 'English'
}

# Load data
data = pd.read_csv('output/multilingual_definitions.csv', delimiter='\t')
data = data[data['confidence']>=0.6]
data['definition_language'] = data['definition_language'].map(lang_map)

# Ignore already translated definitions
output_file = "output/multilingual_definitions_translated.csv"
if os.path.exists(output_file):
    data_existing = pd.read_csv(output_file, delimiter="\t")
    data_existing = data_existing[data_existing['lux_definition'].notna()]
    data = data[~data['code'].isin(data_existing['code'])]

for i, row in data.iterrows():

    messages = [{"role": "user", "content": f"Translate the {row['definition_language']} definition to Luxembourgish. Do not use the original word in Luxembourgish for the definition. Stay as close as possible to the {row['definition_language']} definition, but modify it if the Luxembourgish translation would otherwise sound unnatural. Only return the translation, nothing else.\nLuxembourgish word: {row['lemma']}\n{row['definition_language']} word definition: {row['wn_definition']}"}]

    response = client.responses.create(
        model="gpt-5.2",
        input=messages,
        reasoning={"effort": "none"},
        temperature=0.1,
        ).output_text.strip()

    # response = client.models.generate_content(
    #     model="gemini-3-pro-preview",
    #     contents=messages[0]["content"],
    #     config=types.GenerateContentConfig(
    #         temperature=0.1,
    #         # max_output_tokens=100,
    #         # top_p=0.95,
    #         # top_k=20,
    #         thinking_config=types.ThinkingConfig(thinking_level="low")
    #     )
    # ).text.strip()
    
    lux_definition = response
    
    data.loc[i, 'lux_definition'] = lux_definition

    print('English word: ', row['en_word'])
    print('Luxembourgish word: ', row['lemma'])
    print('English definition: ', row['wn_definition'])
    print('Luxembourgish translation: ', lux_definition)

# Add the new translations to the existing file (if it exists)
if os.path.exists(output_file):
    pd.concat([data_existing, data.dropna(subset=['lux_definition'])], ignore_index=True).to_csv(output_file, sep="\t", index=False)
else:
    data.dropna(subset=['lux_definition']).to_csv(output_file, sep="\t", index=False)
