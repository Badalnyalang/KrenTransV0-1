"""
inference.py
Inference and submission file generation for all 11 language pairs
KrenTransV0-1 | MWire Labs
"""

import os
import sys
import torch
import pandas as pd
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    NllbTokenizer,
)

sys.path.insert(0, "/workspace/IndicTransToolkit/IndicTransToolkit")
from processor import IndicProcessor

# Config
TEAM_NAME = "KrenTransV0-1"
OUTPUT_DIR = "submissions"
TEST_DATA_DIR = "data/test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INDICTRANS_LANGS = {
    "as":       {"tgt_lang": "asm_Beng", "ckpt": "checkpoints/it2_as/best"},
    "brx":      {"tgt_lang": "brx_Deva", "ckpt": "checkpoints/it2_brx/best"},
    "mni":      {"tgt_lang": "mni_Beng", "ckpt": "checkpoints/it2_mni/best"},
    "mni_Mtei": {"tgt_lang": "mni_Mtei", "ckpt": "checkpoints/it2_mni_Mtei/best"},
}

NLLB_LANGS = {
    "lus": {"tgt_lang": "lus_Latn", "ckpt": "checkpoints/nllb_lus/best"},
    "kha": {"tgt_lang": "kha_Latn", "ckpt": "checkpoints/nllb_kha/best"},
    "njz": {"tgt_lang": "njz_Latn", "ckpt": "checkpoints/nllb_njz/best"},
    "trp": {"tgt_lang": "trp_Latn", "ckpt": "checkpoints/nllb_trp/best"},
}

M2M_LANGS = {
    "nag": {"ckpt": "checkpoints/m2m_nag/best"},
    "tgj": {"ckpt": "checkpoints/m2m_tgj/best"},
    "mjw": {"ckpt": "checkpoints/m2m_mjw/best"},
}

NEW_LANGS = ["nag", "tgj", "mjw"]


def load_test(lang):
    path = os.path.join(TEST_DATA_DIR, f"en-{lang}_Test.xlsx")
    df = pd.read_excel(path)
    return df["English Sentences"].tolist()


def save_submission(preds, lang, submission_type="primary"):
    fname = f"{TEAM_NAME}_{submission_type}_en_to_{lang}.txt"
    fpath = os.path.join(OUTPUT_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("\n".join(preds))
    print(f"Saved {fpath} ({len(preds)} lines)")


def run_indictrans(lang, cfg):
    print(f"\nRunning inference: {lang}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["ckpt"], trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["ckpt"], trust_remote_code=True).cuda()
    ip = IndicProcessor(inference=True)
    sentences = load_test(lang)
    preds = []
    for src in sentences:
        processed = ip.preprocess_batch([src], src_lang="eng_Latn", tgt_lang=cfg["tgt_lang"])
        inputs = tokenizer(processed, return_tensors="pt", truncation=True, max_length=128).to("cuda")
        ip.preprocess_batch([src], src_lang="eng_Latn", tgt_lang=cfg["tgt_lang"])
        out = model.generate(**inputs, num_beams=4, max_length=128, repetition_penalty=1.8, no_repeat_ngram_size=3)
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        postprocessed = ip.postprocess_batch([decoded], lang=cfg["tgt_lang"])[0]
        preds.append(postprocessed)
    save_submission(preds, lang)
    del model
    torch.cuda.empty_cache()


def run_nllb(lang, cfg):
    print(f"\nRunning inference: {lang}")
    tokenizer = NllbTokenizer.from_pretrained(cfg["ckpt"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["ckpt"]).cuda()
    forced_bos = tokenizer.convert_tokens_to_ids(cfg["tgt_lang"])
    sentences = load_test(lang)
    preds = []
    tokenizer.src_lang = "eng_Latn"
    for src in sentences:
        inputs = tokenizer(str(src), return_tensors="pt", truncation=True, max_length=128).to("cuda")
        out = model.generate(**inputs, forced_bos_token_id=forced_bos, num_beams=4, max_length=128, repetition_penalty=1.8, no_repeat_ngram_size=3)
        preds.append(tokenizer.decode(out[0], skip_special_tokens=True))
    save_submission(preds, lang)
    del model
    torch.cuda.empty_cache()


def run_m2m(lang, cfg):
    print(f"\nRunning inference: {lang}")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    tokenizer.add_special_tokens({"additional_special_tokens": [f"__{l}__" for l in NEW_LANGS]})
    token_id = tokenizer.convert_tokens_to_ids(f"__{lang}__")
    model = M2M100ForConditionalGeneration.from_pretrained(cfg["ckpt"])
    model.resize_token_embeddings(len(tokenizer))
    model.generation_config.forced_bos_token_id = token_id
    model = model.cuda()
    sentences = load_test(lang)
    preds = []
    tokenizer.src_lang = "en"
    for src in sentences:
        inputs = tokenizer(str(src), return_tensors="pt", truncation=True, max_length=128).to("cuda")
        out = model.generate(**inputs, forced_bos_token_id=token_id, num_beams=4, max_length=128, repetition_penalty=1.8, no_repeat_ngram_size=3)
        preds.append(tokenizer.decode(out[0], skip_special_tokens=True))
    save_submission(preds, lang)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for lang, cfg in INDICTRANS_LANGS.items():
        run_indictrans(lang, cfg)
    for lang, cfg in NLLB_LANGS.items():
        run_nllb(lang, cfg)
    for lang, cfg in M2M_LANGS.items():
        run_m2m(lang, cfg)
    print("\nAll submissions generated.")
