import pandas as pd
from datasets import Dataset
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
            src_lang="deu_Latn",
            tgt_lang="ltz_Latn",
            max_length=400)
    data = pd.read_csv("with_definitions.csv", sep="\t")
    dataset = Dataset.from_pandas(data[["wn_definition"]])
    data["lux_definition"] = translator(dataset)
    data.dropna().to_csv("with_lux_definitions.csv", sep="\t", index=False)


if __name__ == "__main__":
    main()
