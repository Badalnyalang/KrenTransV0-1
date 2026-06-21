# KrenTransV0-1

**WMT 2026 Low-Resource Indic MT Shared Task Submission**
MWire Labs, Shillong, Meghalaya, India

---

## Overview

KrenTransV0-1 is MWire Labs' submission to the WMT 2026 Low-Resource Indic Machine Translation Shared Task covering all 11 Northeast Indian language pairs. The system uses a hybrid architecture combining IndicTrans2, NLLB-200 mono fine-tuned models, and M2M-100 fine-tuned models depending on script, data availability, and observed output quality.

---

## Language Pairs

| Language | Code | Model Used | Submission Type |
|---|---|---|---|
| Assamese | as | IndicTrans2 200M fine-tuned | Unconstrained |
| Bodo | brx | IndicTrans2 200M fine-tuned | Unconstrained |
| Meitei (Bengali script) | mni | IndicTrans2 200M fine-tuned | Unconstrained |
| Meitei (Mayek script) | mni_Mtei | IndicTrans2 200M fine-tuned | Unconstrained |
| Mizo | lus | NLLB-200 1.3B mono fine-tuned | Unconstrained |
| Khasi | kha | NLLB-200 1.3B mono fine-tuned | Unconstrained |
| Nyishi | njz | NLLB-200 1.3B mono fine-tuned | Constrained |
| Kokborok | trp | NLLB-200 1.3B mono fine-tuned | Constrained |
| Nagamese | nag | M2M-100 418M fine-tuned | Constrained |
| Tagin | tgj | M2M-100 418M fine-tuned | Constrained |
| Karbi | mjw | M2M-100 418M fine-tuned | Constrained |

---

## Repository Structure

```
KrenTransV0-1/
├── m2m_finetune.py        # M2M-100 fine-tuning for nag, tgj, mjw
├── nllb_finetune.py       # NLLB-200 mono fine-tuning for lus, kha, njz, trp
├── it2_finetune.py        # IndicTrans2 fine-tuning for brx, as, mni, mni_Mtei
├── inference.py           # Inference and submission file generation
├── reranking.py           # BharatGen reranking experiments (negative result)
└── README.md
```

---

## Models

All base models and fine-tuned checkpoints are available on HuggingFace under [Badnyal](https://huggingface.co/Badnyal) and [MWirelabs](https://huggingface.co/MWirelabs).

| Model | HF Link |
|---|---|
| NLLB Khasi mono | Badnyal/kha-en-nllb-v0.4 |
| NLLB Mizo mono | Badnyal/KrenTrans-mono-lus |
| NLLB Nyishi mono | Badnyal/KrenTrans-mono-njz |
| NLLB Kokborok | Badnyal/kokborok-mt |
| IT2 fine-tuned | Badnyal/KrenTrans-IT2ft-en2xx |
| CP2 multilingual baseline | Badnyal/KrenTransV0-1-cp2 |

---

## Key Findings

- Native NLLB language tags favor mono fine-tuning over multilingual training (+1.94 to +4.11 BLEU)
- IT2 fine-tuning causes negative transfer on Meitei Mayek (−14 BLEU); fine-tuned submitted regardless
- M2M-100 fine-tuning recovers coherent output for completely collapsed languages (nag, tgj, mjw)
- BharatGen Param-1B perplexity reranking does not improve MT output for NE languages (negative result)
- Directional asymmetry confirmed for starved languages (njz→en functional, en→njz near collapse)
- lus and kha augmented with publicly available backtranslated data beyond WMT official training set

---

## Training Details

### M2M-100 (nag, tgj, mjw)
- Base model: `facebook/m2m100_418M`
- Custom language tokens added, initialized from mean of Latin-script language embeddings
- Batch size: 8, LR: 5e-4, fp16, warmup: 100 steps
- Inference: num_beams=4, repetition_penalty=1.8, no_repeat_ngram_size=3

### NLLB-200 (lus, kha, njz, trp)
- Base model: `facebook/nllb-200-1.3B`
- Separate mono fine-tune per language
- lus and kha include publicly available backtranslated data
- Inference: num_beams=4, repetition_penalty=1.8, no_repeat_ngram_size=3

### IndicTrans2 (as, brx, mni, mni_Mtei)
- Base model: `ai4bharat/indictrans2-en-indic-dist-200M`
- Fine-tuned on WMT 2026 official training data per language
- IndicTransToolkit for pre/postprocessing

---

## Data

WMT 2026 official training data used for all language pairs. lus and kha additionally include publicly available backtranslated data. No proprietary data used.
Task data: https://github.com/shantipriyap/WMT2026-NE-IndMT-SharedTask

---

## License

CC-BY-4.0

---
