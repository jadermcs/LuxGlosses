import os
from openai import OpenAI
import pandas as pd
from google import genai
from google.genai import types

def generate_luxembourgish_gloss(client, word, disambiguation):

    disambiguation = f" [{disambiguation}]" if disambiguation else ""

    prompt = {
        "text": (
            f"Given the word in Luxembourgish: '{word}'{disambiguation}, generate a definition for the word in Luxembourgish. "
            "Respond only with the definition in Luxembourgish."
        )
    }

    if type(client) == genai.client.Client:
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                #max_output_tokens=100,
                top_p=0.95,
                top_k=20,
                thinking_config=types.ThinkingConfig(thinking_level="low")
            )
        )
        gloss = response.text.strip()

    elif type(client) == OpenAI:
        response = client.responses.create(
        model="gpt-5.2",
        input=[
            {"role": "system", "content": "You are a helpful assistant that writes glosses in Luxembourgish."},
            {"role": "user", "content": prompt["text"]},
        ],
        max_output_tokens=100,
        temperature=0.1,
        reasoning={"effort": "none"},
        )
        gloss = response.output_text.strip()
    
    return gloss


if __name__ == "__main__":
    #client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY_Trux"))
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    output_file = "output/zero_shot_generated_glosses.csv"
    # Load the CSV file
    df = pd.read_csv("output/multilingual_definitions.csv", sep="\t")

    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        df = df[~df["lemma"].isin(df_existing["lemma"])]
    # Assuming the CSV has columns 'word' and 'sentence'
    for idx, row in df.sample(n=1000).iterrows():
        word = row["lemma"]
        disambiguation = row["en_definition"]
        try:
            gloss = generate_luxembourgish_gloss(
                client,
                word,
                disambiguation,
            )
            df.at[idx, "gloss"] = gloss
            print(f"Word: {word}")
            print(f"Gloss in Luxembourgish: {gloss}")
        except Exception as e:
            print(f"An error occurred while processing the word {word}: {e}")

    # Save the updated DataFrame to a new CSV file
    if os.path.exists(output_file):
        final_df = pd.concat([df_existing, df], ignore_index=True)
    else:
        final_df = df
    final_df.dropna(
        subset=["gloss"], inplace=True
    )
    final_df.to_csv(output_file, index=False)
