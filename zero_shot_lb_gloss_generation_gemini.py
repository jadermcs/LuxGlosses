import os

import pandas as pd
from google import genai


def generate_luxembourgish_gloss(client, word, disambiguation, sentence):
    prompt_with_example = (
        f"Given the word in Luxembourgish: '{word}' [{disambiguation}] and the example sentence: '{sentence}', "
        "generate a definition for the word in Luxembourgish. "
        "Respond only with the definition in Luxembourgish."
    )

    prompt_without_example = (
        f"Given the word in Luxembourgish: '{word}' [{disambiguation}], generate a definition for the word in Luxembourgish. "
        "Respond only with the definition in Luxembourgish."
    )

    response_1 = client.models.generate_content(
        model="gemini-3.0-flash",
        contents=prompt_with_example,
        config={
            "max_output_tokens": 100,
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 20,
        },
    )

    response_2 = client.models.generate_content(
        model="gemini-3.0-flash",
        contents=prompt_without_example,
        config={
            "max_output_tokens": 100,
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 20,
        },
    )

    gloss_with_example = response_1.text.strip()
    gloss_without_example = response_2.text.strip()
    return gloss_with_example, gloss_without_example


if __name__ == "__main__":
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    output_file = "data/zero_shot_generated_glosses_gemini.csv"
    # Load the CSV file
    df = pd.read_csv("data/lod_lux_definitions_zeroshot_gemini.csv", sep="\t")

    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        df = df[~df["lemma"].isin(df_existing["lemma"])]
    # Assuming the CSV has columns 'word' and 'sentence'
    for idx, row in df.sample(n=1000).iterrows():
        word = row["lemma"]
        sentence = row["sentence"]
        disambiguation = row["en_definition"]
        try:
            gloss_with_example, gloss_without_example = generate_luxembourgish_gloss(
                client,
                word,
                disambiguation,
                sentence,
            )
            df.at[idx, "gloss_with_example"] = gloss_with_example
            df.at[idx, "gloss_without_example"] = gloss_without_example
            print(f"Word: {word}")
            print(f"Sentence: {sentence}")
            print(f"Gloss in Luxembourgish (with example): {gloss_with_example}")
            print(
                f"Gloss in Luxembourgish (without example): {gloss_without_example}\n"
            )
        except Exception as e:
            print(f"An error occurred while processing the word {word}: {e}")

    # Save the updated DataFrame to a new CSV file
    if os.path.exists(output_file):
        final_df = pd.concat([df_existing, df], ignore_index=True)
    else:
        final_df = df
    final_df.dropna(
        subset=["gloss_with_example", "gloss_without_example"], how="any", inplace=True
    )
    final_df.to_csv(output_file, index=False)
