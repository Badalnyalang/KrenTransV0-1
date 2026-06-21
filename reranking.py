"""
reranking.py
BharatGen Param-1B perplexity-based reranking experiments (negative result)
NE-BERT fluency reranking experiments (negative result)
KrenTransV0-1 | MWire Labs
"""

import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    NllbTokenizer,
)
from sacrebleu.metrics import BLEU

# Config
NEBERT_MODEL = "MWirelabs/ne-bert"
BHARATGEN_MODEL = "BharatGen/Param-1B"
NLLB_KHA_CKPT = "Badnyal/kha-en-nllb-v0.4"
TEST_DATA = "data/splits/kha_test.tsv"  # 100 held-out samples


def get_nebert_perplexity(text, tokenizer, model, max_len=128):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return outputs.loss.item()


def get_bharatgen_perplexity(text, tokenizer, model, max_len=128):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return outputs.loss.item()


def rerank_with_model(sentences, mt_model, mt_tokenizer, reranker_tokenizer,
                      reranker_model, forced_bos_id, n_best=4):
    preds = []
    for src in sentences:
        mt_tokenizer.src_lang = "eng_Latn"
        inputs = mt_tokenizer(str(src), return_tensors="pt", truncation=True, max_length=128).to("cuda")
        outputs = mt_model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_id,
            num_beams=n_best,
            num_return_sequences=n_best,
            max_length=128,
            repetition_penalty=1.8,
            no_repeat_ngram_size=3,
        )
        hypotheses = [mt_tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        scored = [(h, get_nebert_perplexity(h, reranker_tokenizer, reranker_model))
                  for h in hypotheses]
        best = min(scored, key=lambda x: x[1])[0]
        preds.append(best)
    return preds


def run_nebert_reranking():
    print("Loading NE-BERT reranker...")
    reranker_tokenizer = AutoTokenizer.from_pretrained(NEBERT_MODEL)
    reranker_model = AutoModelForMaskedLM.from_pretrained(NEBERT_MODEL).cuda()
    reranker_model.eval()

    print("Loading NLLB Khasi model...")
    mt_tokenizer = NllbTokenizer.from_pretrained(NLLB_KHA_CKPT)
    mt_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_KHA_CKPT).cuda()
    forced_bos_id = mt_tokenizer.convert_tokens_to_ids("kha_Latn")

    test_df = pd.read_csv(TEST_DATA, sep="\t")
    sentences = test_df["en"].tolist()
    refs = test_df["xx"].tolist()

    # Baseline
    print("Running baseline...")
    baseline_preds = []
    mt_tokenizer.src_lang = "eng_Latn"
    for src in sentences:
        inputs = mt_tokenizer(str(src), return_tensors="pt", truncation=True, max_length=128).to("cuda")
        out = mt_model.generate(**inputs, forced_bos_token_id=forced_bos_id, num_beams=4, max_length=128, repetition_penalty=1.8, no_repeat_ngram_size=3)
        baseline_preds.append(mt_tokenizer.decode(out[0], skip_special_tokens=True))

    # Reranked
    print("Running NE-BERT reranking...")
    reranked_preds = rerank_with_model(
        sentences, mt_model, mt_tokenizer,
        reranker_tokenizer, reranker_model, forced_bos_id
    )

    bleu = BLEU()
    print(f"Baseline BLEU:  {bleu.corpus_score(baseline_preds, [refs])}")
    print(f"Reranked BLEU:  {bleu.corpus_score(reranked_preds, [refs])}")
    print("Finding: NE-BERT reranking does not improve over baseline.")

    del reranker_model, mt_model
    torch.cuda.empty_cache()


def run_bharatgen_reranking():
    print("Loading BharatGen Param-1B reranker...")
    reranker_tokenizer = AutoTokenizer.from_pretrained(BHARATGEN_MODEL)
    reranker_model = AutoModelForCausalLM.from_pretrained(BHARATGEN_MODEL).cuda()
    reranker_model.eval()

    print("Loading NLLB Khasi model...")
    mt_tokenizer = NllbTokenizer.from_pretrained(NLLB_KHA_CKPT)
    mt_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_KHA_CKPT).cuda()
    forced_bos_id = mt_tokenizer.convert_tokens_to_ids("kha_Latn")

    test_df = pd.read_csv(TEST_DATA, sep="\t")
    sentences = test_df["en"].tolist()
    refs = test_df["xx"].tolist()

    print("Running BharatGen reranking...")
    preds = []
    for src in sentences:
        mt_tokenizer.src_lang = "eng_Latn"
        inputs = mt_tokenizer(str(src), return_tensors="pt", truncation=True, max_length=128).to("cuda")
        outputs = mt_model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_id,
            num_beams=4,
            num_return_sequences=4,
            max_length=128,
            repetition_penalty=1.8,
            no_repeat_ngram_size=3,
        )
        hypotheses = [mt_tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        scored = [(h, get_bharatgen_perplexity(h, reranker_tokenizer, reranker_model))
                  for h in hypotheses]
        best = min(scored, key=lambda x: x[1])[0]
        preds.append(best)

    bleu = BLEU()
    print(f"BharatGen Reranked BLEU: {bleu.corpus_score(preds, [refs])}")
    print("Finding: BharatGen Param-1B reranking does not improve over baseline.")

    del reranker_model, mt_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=== NE-BERT Reranking Experiment ===")
    run_nebert_reranking()
    print("\n=== BharatGen Param-1B Reranking Experiment ===")
    run_bharatgen_reranking()
