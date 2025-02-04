import spellux


if __name__ == "__main__":

    with open("data/misspell.txt") as fin:
        for line in fin.readlines():
            line = line.rstrip()
            correct = spellux.normalize_text(line, stats=False)
            print(correct)

    spellux.update_resources(matchdict=True, unknown=False, reset_matchdict=False)
