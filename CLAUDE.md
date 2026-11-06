# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## State of the repo

This directory is the workspace for a master's thesis. Layout:
- `research/research.tex`: Claude's notebook (article summaries, regulatory notes, ideas), mostly written by Claude, in English. Not part of the final paper.
- `articles/pdfs/`: source paper PDFs. `uv run --no-project --with 'pymupdf4llm<2' python articles/pdf2md.py [pdf ...]` converts new PDFs (pymupdf4llm, no OCR; v2.0 on PyPI is an empty stub, so keep `<2`) to `articles/md/<name>.md`; read the .md, not the PDF. `python3 articles/md2bib.py` (also run automatically after pdf2md converts something) **overwrites `work/bibliografia.bib`** from the .md files: lookup by title (Crossref, then arXiv), cite key = first author surname + year (`kirchenbauer2023`, duplicates get `b`, `c`). Papers with no online match live in `articles/manual.bib` (appended as is); the few garbled titles are overridden in `TITLES` in md2bib.py. Never edit `work/bibliografia.bib` by hand, edit those instead. Output is raw (author blocks, two-column artifacts may be imperfect).
- `work/`: the user's final thesis (Polish). Only touch when asked about the thesis itself.

- `src/`: all code lives here. Packages: `src/data/` (all data-related modules), `src/baseline/` (baseline models and their eval/plot scripts), `src/finetune/` (training). Scripts add `src/` to `sys.path` and import as `from data.x import ...`, `from baseline.x import ...`. Run from the repo root as `uv run python src/<pkg>/<file>.py`. Never write code outside `src/`. Install packages only with `uv add` or `pacman`.
  - `src/data/openturingbench.py`: **default dataset**, OpenTuringBench (`MLNTeam-Unical/OpenTuringBench`, config `in_domain`, splits train/val/test; news, 7 open LLMs vs human). `otb_loader(split)` yields batches of (urls, texts, labels, models), label 0 = human / 1 = machine, deduped by text. `url` is shared by the human text and its 7 machine versions (not unique per item). The HF repo has NO human texts: `src/data/fetch_otb_human.py [split ...]` scrapes them from HuffPost (authors' URL list `human_urls.json`, huffingtonpost.com/entry/<slug>, 4 threads, resumable) into `data/openturingbench/human_<split>.jsonl`. Human texts get the same header as machine texts (`title\n\nMonth D, YYYY\n\nbody`) so a detector cannot use the header; the authors' extractor omits it. `src/data/check_format.py` prints per-class format cues (header, quotes, whitespace, placeholders) to verify this. Old header-less test scrape: `data/openturingbench/old_no_header/`.
  - `src/data/opai_bench.py`: older dataset OpAI-Bench (`OpAI-Bench1/OpAI-Bench`, splits train/dev/test), used by `src/finetune/`. `pure_loader(split)` (fully human/AI texts) and `version_loader` (edit version 0..8).
  - `src/baseline/desklib.py`: `desklib_detect(text) -> float`, P(AI-generated) from the HF baseline detector `desklib/ai-text-detector-v1.01`; the thesis baseline.
  - `src/baseline/eval_desklib.py`: desklib on `otb_loader("test")`; writes metrics JSON (incl. per-generator TPR) + per-text scores CSV (`url, model, label, score`) to `data/desklib_openturingbench_test_*`.
  - `src/baseline/plot_desklib.py`: reads those `data/` files and writes ROC / confusion matrix / score histogram PNGs to `graphs/`.
  - `src/finetune/train_desklib.py`: freezes desklib except top 4 layers, swaps head for `Linear(1024, 9)`, trains with cross-entropy to predict OpAI-Bench edit version v0 (human)..v8 (`version_loader`, balanced per class; sizes are constants at the top). Writes metrics to `data/`, confusion PNG to `graphs/`, trained weights to `checkpoints/` (gitignored). ~20 min at 2000/class on RTX 4060.
  - `src/baseline/giga_check.py`: `gigacheck_detect(text) -> float`, P(AI-generated) from `iitolstykh/GigaCheck-Classifier-Multi` (Mistral-7B based, `trust_remote_code`, bf16, ~15 GB, needs transformers 4.x; does not fit the 8 GB GPU, not evaluated).

There is no build system or test suite yet. When more code is added (experiments with small language models, watermark detectors, attack scripts), update this file with the real build/run/test commands and architecture rather than guessing.

The thesis is written in Polish (original topic text is Polish); working notes with Claude may be in English. Ask which language a given file should be in if unclear.

## Thesis topic

**Identifying the origin of LLM-generated text by combining watermark signals and model-fingerprint features, with a probabilistic decision model, evaluated for robustness under adversarial transformations.**

Motivation: LLM text is increasingly indistinguishable from human text (disinformation, fraud, misuse), and the EU AI Act requires marking/detectability of AI-generated content.

Three pillars:
1. **Identification signals**: watermarks and model fingerprinting features; study each alone and combined.
2. **Probabilistic decision model**: specialized model(s) giving a direct, probabilistic judgement of a text's origin (fusion of signals).
3. **Adversarial robustness**: deliberate text transformations that weaken or remove origin information. Measure identification before vs. after attacks, per-signal impact, and ways to harden the whole pipeline. Reliability against deliberate evasion is the main point, not just baseline accuracy.

Planned experiments use **small language models** (open-weight, so watermark access is possible).

## Research agenda (to be covered)

1. **Regulatory**: AI Act Art. 50(2) (machine-readable marking and detectability), Code of Practice on AI-content marking (check current status), C2PA and why metadata does not apply to plain text.
2. **Text watermarking**: logit-based (KGW, Unigram-Watermark, SWEET, EWD); distortion-free/sampling (Aaronson/Kuditipudi, SynthID-Text); semantic/paraphrase-robust (SemStamp, SIR, k-SemStamp); multi-bit and publicly detectable; tooling (MarkLLM, WaterBench).
3. **Fingerprinting/attribution**: passive (stylometry, perplexity/log-prob features, Binoculars, DetectGPT/Fast-DetectGPT, DNA-GPT); source-model attribution ("idiosyncrasies in LLMs"); active fingerprints (Instructional Fingerprinting); watermark radioactivity (Sander et al.).
4. **Fusion/decision model**: calibration (Platt/isotonic, conformal), LR/GBM on features vs. fine-tuned transformer vs. Bayesian log-likelihood-ratio combination; open-set handling (human/unknown-model class). Metrics: AUROC, TPR@1% and 0.1% FPR, ECE. False positive rate is the metric that matters most.
5. **Attacks**: paraphrasing (DIPPER, LLM rewriting, recursive); lexical (synonyms, typos, homoglyphs, invisible chars, emoji attack); round-trip translation; human+AI mixing, partial edits, truncation; watermark stealing/spoofing/scrubbing (Jovanović et al., piggyback); detector evasion (adversarial prompting, DPO-trained evaders, "humanizer" tools). Model attacker as white-box vs. black-box.
6. **Robustness improvement**: adversarial training on attack outputs; ensemble watermark + fingerprint (which signal survives which attack); robustness/quality/detectability trade-offs and theoretical limits (Sadasivan et al.; Zhang et al., "watermarks in the sand").
7. **Experimental design**: datasets RAID, M4/M4GL, HC3, MAGE, Ghostbuster, plus Polish data; open-weight sources (Llama, Qwen, Gemma, Mistral) for watermarking, closed models fingerprint-only; protocol = before/after attack, per-signal and fused, per text length, per language, under domain shift.

## Working conventions

- For literature claims, cite the actual paper (authors, year, venue) and flag anything unverified; do not invent references or numbers. Verify the current status of regulatory items (AI Act, Code of Practice) via search rather than from memory.
- Report detection results with TPR at low FPR and calibration, not accuracy alone.

## Report after writing code

Every time you add or change code, finish with a report so the user does not lose details of what was written:

1. **How it works (technical)**: precise, step by step. Cover the data flow (inputs, transformations, outputs, shapes/types), the key functions and what each does, non-obvious choices (constants, seeds, thresholds, splits, dedup, label mapping), and which files are read/written. Include details that would be needed to reproduce or modify the code later; do not summarize vaguely.
2. **Bug risks**: list concrete ways the code or data could be wrong. Only bugs and flaws count: crashes, silent wrong results, off-by-one or indexing errors, label/ID mix-ups, train/test leakage, flawed or mis-deduped datasets, wrong metric computation, nondeterminism, resource/memory failures, stale cached files. Do NOT list expected behaviour or modelling concerns (model learning poorly, weak accuracy, overfitting as a result of design). For each risk, say what was checked and what was not. If nothing found, say so explicitly.
