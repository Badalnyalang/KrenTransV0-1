"""
m2m_finetune.py
M2M-100 418M fine-tuning for Nagamese (nag), Tagin (tgj), Karbi (mjw)
KrenTransV0-1 | MWire Labs
"""

import os
import torch
import gc
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Config
DATA_PATHS = {
    "nag": "data/nag.tsv",
    "tgj": "data/tgj.tsv",
    "mjw": "data/mjw.tsv",
}
OUTPUT_DIR = "checkpoints"
MODEL_NAME = "facebook/m2m100_418M"
NEW_LANGS = ["nag", "tgj", "mjw"]
LATIN_LANGS = [
    "ff", "ilo", "af", "cy", "da", "de", "en", "es", "et", "fi", "fr",
    "ga", "gl", "hr", "ht", "hu", "id", "it", "lb", "lt", "lv", "mg",
    "ms", "nl", "no", "oc", "pl", "pt", "ro", "sk", "sl", "sq", "ss",
    "su", "sv", "sw", "tl", "tn", "tr", "uz", "vi", "wo", "xh", "yo", "zu"
]

class MTDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src = str(self.df.iloc[idx]["en"])
        tgt = str(self.df.iloc[idx]["xx"])
        self.tokenizer.src_lang = "en"
        model_inputs = self.tokenizer(src, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
        self.tokenizer.src_lang = "ff"  # dummy for target tokenization
        labels = self.tokenizer(tgt, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
        model_inputs["labels"] = labels["input_ids"]
        return {k: v.squeeze() for k, v in model_inputs.items()}


def load_and_split(path, test_size=100):
    df = pd.read_csv(path, sep="\t")
    train, test = train_test_split(df, test_size=test_size, random_state=42)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def init_model_and_tokenizer():
    tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
    model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": [f"__{l}__" for l in NEW_LANGS]})
    model.resize_token_embeddings(len(tokenizer))
    latin_ids = [tokenizer.lang_code_to_id[l] for l in LATIN_LANGS]
    with torch.no_grad():
        mean_embed = model.model.shared.weight[latin_ids].mean(dim=0)
        for lang in NEW_LANGS:
            token_id = tokenizer.convert_tokens_to_ids(f"__{lang}__")
            model.model.shared.weight[token_id] = mean_embed
    return model, tokenizer


def train_language(lang, model, tokenizer, train_df, epochs=5, batch_size=8):
    token_id = tokenizer.convert_tokens_to_ids(f"__{lang}__")
    model.generation_config.forced_bos_token_id = token_id
    model = model.cuda()
    dataset = MTDataset(train_df, tokenizer)
    args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"m2m_{lang}"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="epoch",
        fp16=True,
        learning_rate=5e-4,
    )
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=dataset)
    trainer.train()
    print(f"{lang} training done.")
    return model


if __name__ == "__main__":
    model, tokenizer = init_model_and_tokenizer()
    for lang, path in DATA_PATHS.items():
        print(f"\nTraining {lang}...")
        train_df, _ = load_and_split(path)
        model, tokenizer = init_model_and_tokenizer()  # fresh model per language
        model = train_language(lang, model, tokenizer, train_df)
        del model
        torch.cuda.empty_cache()
        gc.collect()
