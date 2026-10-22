# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## State of the repo

This directory is the workspace for a master's thesis. Layout:
- `research/research.tex`: Claude's notebook (article summaries, regulatory notes, ideas), mostly written by Claude, in English. Not part of the final paper.
- `articles/pdfs/`: source paper PDFs. `python3 articles/pdf2md.py [pdf ...]` converts them (pymupdf4llm) to `articles/md/<name>.md`; read the .md, not the PDF. Output is raw (author blocks, two-column artifacts may be imperfect).
- `work/`: the user's final thesis (Polish). Only touch when asked about the thesis itself.

- `src/`: all code lives here, as flat modules directly in `src/` (no package subdirectory). Never write code elsewhere.
  - `src/desklib.py`: `desklib_detect(text) -> float`, P(AI-generated) from the HF baseline detector `desklib/ai-text-detector-v1.01`, for comparison with the local model. Run: `uv run python src/desklib.py`.

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
