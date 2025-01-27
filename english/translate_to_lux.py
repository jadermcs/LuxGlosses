import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline


def main():
    checkpoint = "facebook/nllb-200-3.3B"
    model_nllb = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer_nllb = AutoTokenizer.from_pretrained(checkpoint)
    translator = pipeline(
            "translation",
            model=model_nllb,
            tokenizer=tokenizer_nllb,
            src_lang="eng_Latn",
            tgt_lang="ltz_Latn",
            max_length=400)
    data = pd.read_csv("data/english_definitions.csv", sep="\t")
    data = data[data.confidence > 0.8]
    tqdm.pandas(desc="Analyzing sentiment")
    data["lux_definition"] = data["wn_definition"].progress_apply(lambda x: translator(x)[0]["translation_text"])

    data.dropna().to_csv("data/lod_lux_definitions.csv", sep="\t", index=False)


if __name__ == "__main__":
    main()
