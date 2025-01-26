import nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.util import bigrams
from symspellpy import SymSpell, Verbosity
from collections import Counter


nltk.download('punkt')


def build_bigram_dict(corpus):
    counter1 = Counter()
    counter2 = Counter()
    for line in tqdm(corpus):
        words = word_tokenize(line)
        counter1.update(words)
        bigram_list = bigrams(words)
        counter2.update(bigram_list)
    return counter1, counter2


def lux_spell(uni_path, bi_path):
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym_spell.load_dictionary(uni_path, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bi_path, term_index=0, count_index=2)
    return sym_spell


def generate_dictionaries(uni, bi):
    with open("sentences.csv") as fin:
        data = fin.readlines()
    # from datasets import load_dataset
    # dataset = load_dataset("HuggingFaceFW/fineweb-2", name="ltz_Latn")
    # data = dataset["train"]["text"]

    count1, count2 = build_bigram_dict(data)

    with open(uni, "w") as fout:
        for a, c in count1.items():
            fout.write(f"{a} {c}\n")

    with open(bi, "w") as fout:
        for (a, b), c in count2.items():
            fout.write(f"{a} {b} {c}\n")


if __name__ == "__main__":
    uni_path = "data/unigram_count.txt"
    bi_path = "data/bigram_count.txt"

    generate_dictionaries(uni_path, bi_path)

    checker = lux_spell(uni_path, bi_path)

    examples = [
            "eng Kammer, an där de Kolben bewegt"
            ]
    print("Simple checker:")
    for example in examples:
        print("Original:", example)
        print("Corrected:", end=" ")
        for word in word_tokenize(example):
            suggestions = checker.lookup(
                word,
                Verbosity.TOP,
                max_edit_distance=2,
                transfer_casing=True,
                )
            for sugg in suggestions:
                print(sugg.term, end=" ")
        print()
    print("Compound checker:")
    for example in examples:
        suggestions = checker.lookup_compound(
            example,
            max_edit_distance=2,
            transfer_casing=True,
            )
        print("Original:", example)
        print("Corrected:", suggestions[0].term)
