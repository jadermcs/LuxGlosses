from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import evaluate
import torch
from transformers import BitsAndBytesConfig

config = BitsAndBytesConfig(
    load_in_8bit=True,
)

# Load model and tokenizer
model_name = "NousResearch/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=config,
        attn_implementation="sdpa",
        torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


# Configure LoRA
lora_config = LoraConfig(
    r=16,  # Low-rank matrix dimension
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"  # Task type for language modeling
)

# Wrap the model with LoRA
model = get_peft_model(model, lora_config)


# Load your dataset
dataset = load_dataset("HuggingFaceFW/fineweb-2", name="ltz_Latn")


# Tokenization function
block_size = 512


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# Apply preprocessing
tokenized_dataset = dataset.map(group_texts, batched=True)


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    eval_strategy="steps",
    save_total_limit=2,
    fp16=True,  # Use mixed precision for faster training
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].select(range(500)),
    eval_dataset=tokenized_dataset["test"],
)

# Fine-tune the model
trainer.train()

dataset = load_dataset(
        "csv",
        data_files="chatgpt_correct.csv",
        delimiter="\t"
        ).select_columns(["lemma", "sentence", "lux_definition"])
dataset = dataset["train"].train_test_split(test_size=0.3)


def prompt(examples):
    text = (
        "Generate the dictionary definition in Luxembourgish for the "
        + "word '"
        + examples["lemma"]
        + "' in the context \""
        + examples["sentence"]
        + '".'
    )
    return {"prompt": text, "answer": examples["lux_definition"]}


dataset = dataset.map(prompt)


def preprocess_function(examples):
    inputs = tokenizer(
        examples["prompt"] + examples["answer"],
        truncation=True, padding="max_length", max_length=512)
    inputs["labels"] = inputs["input_ids"]
    return inputs


# Apply preprocessing
tokenized_dataset = dataset.map(preprocess_function)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./luxgloss",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    eval_strategy="steps",
    save_total_limit=2,
    fp16=True,  # Use mixed precision for faster training
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Fine-tune the model
trainer.train()
model.save_pretrained("./luxgloss")
model.eval()

bleu = evaluate.load("bleu")


# Function to generate translations
def generate_translations(batch):
    inputs = tokenizer(
            batch["prompt"],
            return_tensors="pt",
            truncation=True,
            padding=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    batch["predicted"] = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return batch


# Generate predictions
eval_data = dataset["test"].map(generate_translations, batched=True)

# Prepare predictions and references
predictions = eval_data["predicted"]
references = [[ref] for ref in eval_data["answer"]]  # BLEU expects references as lists of lists

# Compute BLEU
bleu_score = bleu.compute(predictions=predictions, references=references)
print("BLEU Score:", bleu_score["bleu"])


# Save the LoRA-adapted model

# Evaluate
print("Training completed!")
