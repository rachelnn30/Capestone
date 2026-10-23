"""CISC 473 P08: attempt a full-test-set reproduction of a published Lead-3 baseline.

Reference: See, Liu, Manning (2017), Get To The Point: Summarization with
Pointer-Generator Networks, Table 1: Lead-3 CNN/DailyMail TEST ROUGE F1:
R1=40.34, R2=17.70, RL=36.57.

This runs a matching *task* and full test split, using first three sentences,
but NOT the paper's exact pyrouge/Perl ROUGE-1.5.5 evaluation pipeline.
It also uses a rule-based Punkt sentence splitter rather than pretrained Punkt. It is an
approximate reproduction, not proof of protocol-exact reproduction.

Run inside the existing P08_Capstone virtual environment:
    python -m pip install nltk
    python P08_lead3_no_download.py

The script saves an independent file in results/ and DOES NOT overwrite your
existing pilot results or train any models.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset
from rouge_score import rouge_scorer

# Avoid NLTK's model-data downloader, which can fail on macOS SSL certificates.
# This is a rule-based Punkt tokenizer with common English abbreviations.
# It is NOT the pretrained English Punkt model; therefore the result is an
# approximate reproduction and must not be described as an exact match.
try:
    from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
except ImportError as exc:
    raise SystemExit("Install NLTK first: python -m pip install nltk") from exc

_params = PunktParameters()
_params.abbrev_types = {
    "mr", "mrs", "ms", "dr", "prof", "sen", "rep", "gov", "jr", "sr",
    "st", "vs", "etc", "inc", "corp", "u.s", "u.k", "no", "fig",
}
_sentence_tokenizer = PunktSentenceTokenizer(_params)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

PAPER = {"rouge1": 40.34, "rouge2": 17.70, "rougeLsum": 36.57}
METRICS = ("rouge1", "rouge2", "rougeL", "rougeLsum")


def sentences(text: str) -> list[str]:
    return [part.strip() for part in _sentence_tokenizer.tokenize(text) if part.strip()]


def main() -> None:
    print("Using built-in rule-based Punkt splitter (NO downloads needed).", flush=True)
    print("Note: sentence boundaries may differ from the pretrained Punkt / original paper.", flush=True)
    print("Loading official CNN/DailyMail 3.0.0 FULL test split...", flush=True)
    test = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
    print(f"Loaded {len(test):,} test articles. No model training is needed.", flush=True)

    scorer = rouge_scorer.RougeScorer(METRICS, use_stemmer=True)
    total_f1 = {k: 0.0 for k in METRICS}
    count = 0

    for example in test:
        # First three article sentences = the published Lead-3 method.
        # Newline separates summary sentences so rougeLsum is calculated
        # as a summary-level LCS, closer to classic multi-sentence ROUGE-L.
        lead3 = "\n".join(sentences(example["article"])[:3])
        reference = "\n".join(sentences(example["highlights"]))
        if not lead3 or not reference:
            continue
        per_item = scorer.score(reference, lead3)
        for name in METRICS:
            total_f1[name] += per_item[name].fmeasure
        count += 1
        if count % 500 == 0:
            print(f"Scored {count:,}/{len(test):,} test articles...", flush=True)

    if count == 0:
        raise RuntimeError("No valid examples were scored.")

    ours = {k: 100 * total_f1[k] / count for k in METRICS}
    print("\n===== FULL-TEST LEAD-3 COMPARISON (percentage points) =====", flush=True)
    print(f"Dataset: CNN/DailyMail 3.0.0, test split, {count:,} examples")
    print("Source: See et al. (2017), Table 1; published results use pyrouge.")
    print("Our evaluation: python rouge_score, stemming ON; ROUGE-Lsum approximates paper's ROUGE-L.")
    print("Sentence splitter: locally configured rule-based Punkt, not pretrained Punkt.")
    print(f"{'Metric':<18} {'Published':>12} {'Ours':>12} {'Difference':>13} {'Rel diff':>12}")

    comparison = {}
    for metric, label in (("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"), ("rougeLsum", "ROUGE-Lsum vs L")):
        paper = PAPER[metric]
        local = ours[metric]
        diff = local - paper
        relative = abs(diff) / paper * 100
        print(f"{label:<18} {paper:12.2f} {local:12.2f} {diff:+13.2f} {relative:11.2f}%")
        comparison[metric] = {
            "published_percent": paper,
            "local_percent": local,
            "difference_in_rouge_points": diff,
            "absolute_relative_error_percent": relative,
            "within_two_rouge_points": abs(diff) <= 2.0,
            "within_two_percent_relative": relative <= 2.0,
        }

    print(f"\nExtra diagnostic: rougeL (not rougeLsum) = {ours['rougeL']:.2f}")
    print("\nIMPORTANT: A numerical match is not necessarily an exact reproduction:")
    print("the paper uses the original dataset preparation and official pyrouge scoring;")
    print("this run uses the Hugging Face dataset and Python rouge_score.")
    print("Describe it as an approximate full-test baseline reproduction.")

    payload = {
        "dataset": "abisee/cnn_dailymail, 3.0.0",
        "split": "test",
        "n_scored": count,
        "baseline": "Lead-3 first 3 article sentences",
        "published_reference": "See, Liu, Manning (2017), Table 1",
        "evaluation": "rouge_score.RougeScorer, Porter stemming, newline sentence boundaries; mean F1",
        "sentence_splitter": "Rule-based NLTK PunktSentenceTokenizer with common-abbreviation list; no pretrained data",
        "not_exact_sentence_segmenter": True,
        "not_exact_published_pyrouge": True,
        "published": PAPER,
        "our_scores_percent": ours,
        "comparisons": comparison,
    }
    outpath = RESULTS / "lead3_full_test_reproduction.json"
    outpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {outpath}", flush=True)


if __name__ == "__main__":
    main()
