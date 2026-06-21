"""
it2_finetune.py
IndicTrans2 200M fine-tuning for Assamese (as), Bodo (brx), Meitei Bengali (mni), Meitei Mayek (mni_Mtei)
KrenTransV0-1 | MWire Labs
"""

import os
import sys
import torch
import gc
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Add IndicTransToolkit to path
sys.path.insert(0, "/workspace/IndicTransToolkit/IndicTransToolkit")
from processor import IndicProcessor

# Config
MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
OUTPUT_DIR = "checkpoints"

LANG_CONFIG = {
    "as":      {"tgt_lang": "asm_Beng", "data": "data/as.tsv"},
    "brx":     {"tgt_lang": "brx_Deva", "data": "data/brx.tsv"},
    "mni":     {"tgt_lang": "mni_Beng", "data": "data/mni.tsv"},
    "mni_Mtei":{"tgt_lang": "mni_Mtei", "data": "data/mni_Mtei.tsv"},
}


class IT2Dataset(Dataset):
    def __init__(self, df, tokenizer, ip, tgt_lang, max_len=128):
        self.df = df
        self.tokenizer = tokenizer
        self.ip = ip
        self.tgt_lang = tgt_lang
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src = str(self.df.iloc[idx]["en"])
        tgt = str(self.df.iloc[idx]["xx"])

        src_processed = self.ip.preprocess_batch([src], src_lang="eng_Latn", tgt_lang=self.tgt_lang)[0]
        tgt_processed = self.ip.preprocess_batch([tgt], src_lang=self.tgt_lang, tgt_lang="eng_Latn")[0]

        model_inputs = self.tokenizer(
            src_processed, max_length=self.max_len, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_processed, max_length=self.max_len, truncation=True,
                padding="max_length", return_tensors="pt"
            )
        model_inputs["labels"] = labels["input_ids"]
        return {k: v.squeeze() for k, v in model_inputs.items()}


def load_and_split(path, test_size=100):
    df = pd.read_csv(path, sep="\t")
    train, test = train_test_split(df, test_size=test_size, random_state=42)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def train_language(lang, cfg):
    print(f"\nTraining {lang}...")
    train_df, _ = load_and_split(cfg["data"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    ip = IndicProcessor(inference=False)

    dataset = IT2Dataset(train_df, tokenizer, ip, cfg["tgt_lang"])
    model = model.cuda()

    args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"it2_{lang}"),
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
