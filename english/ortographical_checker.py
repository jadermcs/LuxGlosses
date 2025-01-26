import hunspell
from nltk.tokenize import word_tokenize


def lux_spell(dic_path, aff_path):
    hunchecker = hunspell.HunSpell(dic_path, aff_path)
    return hunchecker


if __name__ == "__main__":

    dic_path = "data/lux.dic"
    aff_path = "data/lux.aff"
    checker = lux_spell(dic_path, aff_path)

    examples = [
            "eng Kammer, an där de Kolben bewegt"
            ]
    for example in examples:
        print("Original:", example)
        print("Corrected:", end=" ")
        for word in word_tokenize(example):
            if checker.spell(word):
                print(word, end=" ")
            else:
                suggestion = checker.suggest(
                    word,
                    )
                print(suggestion[0], end=" ")
