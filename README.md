# LuxGlosses

To set up the environment, run the following commands in the terminal:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 1: Create dataset of Luxembourgish words and their translations
To create a dataset of Luxembourgish words and their respective English, German, and French translations, run the following command:
```bash
python src/sensealign/utils/lod_data.py
```
This will create a file `data/lod_multilingual_words.csv` with the Luxembourgish words and their translations.


### Step 2: Align Luxembourgish words with their respective glosses
To align Luxembourgish words (from `data/lod_multilingual_words.csv`) with their respective English, German, and French glosses, run the following command:
```bash
python src/sensealign/get_multilingual_definitions.py
```
This will create a file `output/multilingual_definitions.csv` with the aligned glosses.


### Step 3: Translate the aligned glosses to Luxembourgish
Then, to translate the aligned glosses to Luxembourgish, run:
```bash
python src/sensealign/translate_definitions_to_lux.py
```
This will create a file `output/multilingual_definitions_translated.csv` with the aligned glosses and their translations in Luxembourgish.


### Step 4: Create a dataset of zero-shot LLM-generated glosses for Luxembourgish words
Finally, to create a dataset of zero-shot LLM-generated glosses for Luxembourgish words, run:
```bash
python src/zero_shot_lb_gloss_generation.py
```
This will create a file `output/zero_shot_glosses.csv` with the generated glosses for Luxembourgish words.


### Step 5: Evaluate the translation quality of the translated glosses
To evaluate the translation quality of the translated glosses (using human post-edited translations), `src/translation_metrics.py` can be used.


### Notes
- Currently, there is still a `old_files` folder, containing some of the files from the previous paper version. These files are not used in the current codebase, but they are kept for reference.