"""CISC 473 P08: three independent LoRA-rank-4 runs and matched pilot baselines.

Run from the P08_Capstone folder:
    python P08_three_seed_experiment.py

Optional smoke test (not for report):
    python P08_three_seed_experiment.py --train-size 4 --val-size 2 --seeds 42
"""

import argparse
import csv
import gc
import json
import os
import re
import statistics
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import evaluate
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

MODEL_ID = "facebook/bart-base"
METRICS = ("rouge1", "rouge2", "rougeL")
RESULTS_DIR = Path(__file__).resolve().parent / "results"

def make_lead3(article):
    # Simple sentence heuristic. This is not necessarily identical to the
    # original paper's preprocessing and ROUGE scoring pipeline.
    sentences = re.split(r"(?<=[.!?])\s+", article.strip())
    return " ".join(sentences[:3])


def predictions(model, tokenizer, dataset, device, max_input, max_summary):
    model.eval()
    out = []
    for idx, item in enumerate(dataset):
        tokens = tokenizer(item["article"], return_tensors="pt", max_length=max_input,
                           truncation=True).to(device)
        with torch.no_grad():
            decoded = model.generate(**tokens, max_new_tokens=max_summary,
                                     num_beams=4, early_stopping=True)
        out.append(tokenizer.decode(decoded[0], skip_special_tokens=True))
        print(f"  Generated {idx + 1}/{len(dataset)}", flush=True)
    return out


def score(rouge, predictions_, references):
    # Use the same metric implementation for all our locally evaluated systems.
    x = rouge.compute(predictions=predictions_, references=references,
                      use_stemmer=False)
    return {metric: float(x[metric]) for metric in METRICS}


def save_progress(rows, path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "seed", "train_examples", "val_examples",
                                                 "rouge1", "rouge2", "rougeL", "train_seconds",
                                                 "trainable_parameters"])
        writer.writeheader()
        writer.writerows(rows)


def pretty(scores):
    return ", ".join(f"{k}={scores[k]:.4f}" for k in METRICS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=20)
    parser.add_argument("--val-size", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-input", type=int, default=256)
    parser.add_argument("--max-summary", type=int, default=64)
    args = parser.parse_args()
    if args.train_size < 1 or args.val_size < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Supply positive sizes and distinct random seeds.")

    RESULTS_DIR.mkdir(exist_ok=True)
    print("Loading CNN/DailyMail train and OFFICIAL validation splits...", flush=True)
    train_raw = load_dataset("abisee/cnn_dailymail", "3.0.0",
                             split=f"train[:{args.train_size}]")
    val_raw = load_dataset("abisee/cnn_dailymail", "3.0.0",
                           split=f"validation[:{args.val_size}]")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    rouge = evaluate.load("rouge")
    references = [r["highlights"] for r in val_raw]

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}; training examples: {len(train_raw)}; validation examples: {len(val_raw)}")

    def preprocess(items):
        inputs = tokenizer(items["article"], truncation=True, max_length=args.max_input)
        summary_tokens = tokenizer(text_target=items["highlights"], truncation=True,
                                   max_length=args.max_summary)
        inputs["labels"] = summary_tokens["input_ids"]
        return inputs

    tokenized = train_raw.map(preprocess, batched=True, remove_columns=train_raw.column_names)

    # Deterministic, untrained pretrained-model baseline.
    print("\nOriginal pretrained BART-base (no task fine-tuning)...", flush=True)
    baseline = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device)
    baseline_texts = predictions(baseline, tokenizer, val_raw, device,
                                 args.max_input, args.max_summary)
    baseline_scores = score(rouge, baseline_texts, references)
    print("Original BART:", pretty(baseline_scores), flush=True)
    del baseline
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    # Lead-3 is an explicitly published, model-free CNN/DailyMail baseline.
    lead3_texts = [make_lead3(r["article"]) for r in val_raw]
    lead3_scores = score(rouge, lead3_texts, references)
    print("Lead-3 (approximate implementation):", pretty(lead3_scores), flush=True)

    rows = []
    for name, scores_ in [("BART-base pretrained", baseline_scores), ("Lead-3 heuristic", lead3_scores)]:
        rows.append(dict(model=name, seed="N/A", train_examples=0,
                         val_examples=len(val_raw), **scores_,
                         train_seconds=0, trainable_parameters=0))
    save_progress(rows, RESULTS_DIR / "all_model_scores.csv")

    all_lora_outputs = {}
    for seed in args.seeds:
        print(f"\n===== LoRA independent run: seed {seed} =====", flush=True)
        set_seed(seed)
        # CRITICAL: build a fresh base and fresh LoRA matrices on EVERY seed.
        fresh_base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
        config = LoraConfig(r=4, lora_alpha=16, lora_dropout=0.1,
                            target_modules=["q_proj", "v_proj"], bias="none",
                            task_type=TaskType.SEQ_2_SEQ_LM)
        model = get_peft_model(fresh_base, config)
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable LoRA parameters: {trainable_count:,}")

        arguments = Seq2SeqTrainingArguments(
            output_dir=str(RESULTS_DIR / f"temporary_seed_{seed}"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=2,
            learning_rate=2e-4,
            optim="adamw_torch",
            logging_steps=2,
            save_strategy="no",
            report_to="none",
            fp16=False,
            dataloader_num_workers=0,
            seed=seed,
            data_seed=seed,
        )
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        trainer = Seq2SeqTrainer(model=model, args=arguments,
                                 train_dataset=tokenized, data_collator=collator)
        start = time.monotonic()
        trainer.train()
        elapsed = time.monotonic() - start
        # Trainer has already moved the model to its device.
        model_device = next(model.parameters()).device
        texts = predictions(model, tokenizer, val_raw, model_device,
                            args.max_input, args.max_summary)
        all_lora_outputs[seed] = texts
        scores_ = score(rouge, texts, references)
        rows.append(dict(model="BART-base + LoRA r=4", seed=seed,
                         train_examples=len(train_raw), val_examples=len(val_raw),
                         **scores_, train_seconds=round(elapsed, 2),
                         trainable_parameters=trainable_count))
        save_progress(rows, RESULTS_DIR / "all_model_scores.csv")
        print(f"Seed {seed}: {pretty(scores_)}; training {elapsed:.1f} seconds", flush=True)
        del trainer, model, fresh_base, collator
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    lora_rows = [r for r in rows if r["model"] == "BART-base + LoRA r=4"]
    summary = {}
    for metric in METRICS:
        values = [row[metric] for row in lora_rows]
        summary[metric] = {"mean": statistics.mean(values),
                           "sample_std": statistics.stdev(values) if len(values) >= 2 else None,
                           "runs": len(values)}
    print("\n===== Results =====")
    print("Published Lead-3 FULL CNN/DM TEST (See et al., 2017): R1=.4034 R2=.1770 RL=.3657")
    print("Our Lead-3 SMALL VALIDATION subset:", pretty(lead3_scores))
    print("Our unfine-tuned BART SMALL VALIDATION subset:", pretty(baseline_scores))
    for metric in METRICS:
        data = summary[metric]
        std_txt = f"{data['sample_std']:.4f}" if data["sample_std"] is not None else "N/A"
        print(f"LoRA r=4 {metric} across {len(args.seeds)} seeds: {data['mean']:.4f} ± {std_txt}")

    example = {
        "dataset": "CNN/DailyMail 3.0.0 official validation split, first example",
        "article_excerpt": val_raw[0]["article"][:550],
        "reference": references[0],
        "bart_base": baseline_texts[0],
        "lead3": lead3_texts[0],
        "lora_seed_outputs": {str(seed): all_lora_outputs[seed][0] for seed in args.seeds},
    }
    payload = {
        "notes": ["All local model scores use same official validation subset.",
                  "Published Lead-3 scores from See et al. (2017) were measured on the full TEST split using a different scoring implementation; do NOT claim exact benchmark reproduction.",
                  "This uses short sequences and a very small dataset; results are a pilot only."],
        "model": MODEL_ID,
        "configuration": vars(args),
        "validation_source": "abisee/cnn_dailymail, 3.0.0, validation[:N]",
        "published_lead3_test_rouge": {"rouge1": 0.4034, "rouge2": 0.1770, "rougeL": 0.3657},
        "baseline_scores": baseline_scores,
        "lead3_scores": lead3_scores,
        "all_rows": rows,
        "three_seed_summary": summary,
        "qualitative_example": example,
    }
    (RESULTS_DIR / "experiment_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS_DIR / "qualitative_example.txt").write_text(
        "ARTICLE EXCERPT:\n" + example["article_excerpt"] + "\n\n"
        + "HUMAN REFERENCE:\n" + example["reference"] + "\n\n"
        + "UNFINE-TUNED BART:\n" + example["bart_base"] + "\n\n"
        + "LEAD-3:\n" + example["lead3"] + "\n\n"
        + "LORA, SEED " + str(args.seeds[0]) + ":\n" + example["lora_seed_outputs"][str(args.seeds[0])] + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
