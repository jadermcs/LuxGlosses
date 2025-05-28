import os
import pandas as pd
from openai import OpenAI

def generate_luxembourgish_gloss(word, sentence):
    prompt_with_example = (
        f"Given the word: '{word}' and the example sentence: '{sentence}', "
        "generate a gloss (definition or explanation) for the word in Luxembourgish. "
        "Respond only with the gloss in Luxembourgish."
    )

    prompt_without_example = (
        f"Given the word: '{word}', generate a gloss (definition or explanation) for the word in Luxembourgish. "
        "Respond only with the gloss in Luxembourgish."
    )

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response_1 = client.chat.completions.create(
        model="gpt-4.5-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that writes glosses in Luxembourgish."},
            {"role": "user", "content": prompt_with_example}
        ],
        max_tokens=100,
        temperature=0.3,
    )

    response_2 = client.chat.completions.create(
        model="gpt-4.5-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that writes glosses in Luxembourgish."},
            {"role": "user", "content": prompt_without_example}
        ],
        max_tokens=100,
        temperature=0.3,
    )

    gloss_with_example = response_1.choices[0].message.content.strip()
    gloss_without_example = response_2.choices[0].message.content.strip()
    return gloss_with_example, gloss_without_example

if __name__ == "__main__":
    # Load the CSV file
    df = pd.read_csv("data/Evaluation LuxGlosses.csv")
    # Assuming the CSV has columns 'word' and 'sentence'
    for idx, row in df.iterrows():
        word = row['lemma']
        sentence = row['sentence']
        gloss_with_example, gloss_without_example = generate_luxembourgish_gloss(word, sentence)
        df.at[idx, 'gloss_with_example'] = gloss_with_example
        df.at[idx, 'gloss_without_example'] = gloss_without_example
        print(f"Word: {word}")
        print(f"Sentence: {sentence}")
        print(f"Gloss in Luxembourgish (with example): {gloss_with_example}")
        print(f"Gloss in Luxembourgish (without example): {gloss_without_example}\n")
    
    # Save the updated DataFrame to a new CSV file
    df.drop(columns=['necessary_edits', 'annotator'], inplace=True)  # Remove the sentence column if not needed
    df.to_csv("data/zero_shot_generated_glosses.csv", index=False)