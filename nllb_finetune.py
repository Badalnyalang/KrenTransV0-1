"""
nllb_finetune.py
NLLB-200 1.3B mono fine-tuning for Mizo (lus), Khasi (kha), Nyishi (njz), Kokborok (trp)
KrenTransV0-1 | MWire Labs
"""

import os
import torch
import gc
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    NllbTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

# Config
MODEL_NAME = "facebook/nllb-200-1.3B"
OUTPUT_DIR = "checkpoints"

LANG_CONFIG = {
    "lus": {"src_lang": "eng_Latn", "tgt_lang": "lus_Latn", "data": "data/lus.tsv"},
    "kha": {"src_lang": "eng_Latn", "tgt_lang": "kha_Latn", "data": "data/kha.tsv"},
    "njz": {"src_lang": "eng_Latn", "tgt_lang": "njz_Latn", "data": "data/njz.tsv"},
    "trp": {"src_lang": "eng_Latn", "tgt_lang": "trp_Latn", "data": "data/trp.tsv"},
}

CUSTOM_LANG_TOKENS = ["njz_Latn", "trp_Latn"]  # not native to NLLB


class MTDataset(Dataset):
    def __init__(self, df, tokenizer, src_lang, tgt_lang, max_len=128):
        self.df = df
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src = str(self.df.iloc[idx]["en"])
        tgt = str(self.df.iloc[idx]["xx"])
        self.tokenizer.src_lang = self.src_lang
        model_inputs = self.tokenizer(
            src, max_length=self.max_len, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        self.tokenizer.src_lang = self.tgt_lang
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt, max_length=self.max_len, truncation=True,
                padding="max_length", return_tensors="pt"
            )
        model_inputs["labels"] = labels["input_ids"]
        return {k: v.squeeze() for k, v in model_inputs.items()}


def load_and_split(path, test_size=100):
    df = pd.read_csv(path, sep="\t")
    train, test = train_test_split(df, test_size=test_size, random_state=42)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def init_model_and_tokenizer(custom_tokens=None):
    tokenizer = NllbTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    if custom_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": custom_tokens})
        model.resize_token_embeddings(len(tokenizer))
        # Initialize custom tokens from mean of Latin-script NLLB tokens
        latin_ids = [
            tokenizer.convert_tokens_to_ids(t)
            for t in tokenizer.additional_special_tokens
            if "Latn" in t and t not in custom_tokens
        ]
        with torch.no_grad():
            mean_embed = model.model.shared.weight[latin_ids].mean(dim=0)
            for token in custom_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token)
                model.model.shared.weight[token_id] = mean_embed
    return model, tokenizer


def train_language(lang, cfg):
    print(f"\nTraining {lang}...")
    train_df, _ = load_and_split(cfg["data"])
    custom = [cfg["tgt_lang"]] if cfg["tgt_lang"] in CUSTOM_LANG_TOKENS else None
    model, tokenizer = init_model_and_tokenizer(custom_tokens=custom)
    dataset = MTDataset(train_df, tokenizer, cfg["src_lang"], cfg["tgt_lang"])
    forced_bos = tokenizer.convert_tokens_to_ids(cfg["tgt_lang"])
    model.generation_config.forced_bos_token_id = forced_bos
    model = model.cuda()
    args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"nllb_{lang}"),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        warmup_steps=200,
        weight_decay=0.01,
        logging_steps=200,
        save_strategy="epoch",
        fp16=True,
        learning_rate=3e-4,
    )
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=dataset)
    trainer.train()
    print(f"{lang} done.")
    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    for lang, cfg in LANG_CONFIG.items():
        train_language(lang, cfg)
