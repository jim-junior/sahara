#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm

# ───────────────────────── stdlib ────────────────────────────
import os
import time
import json
import argparse
import logging
import warnings
import asyncio
import ast
import random
# ──────────────────────── third-party ────────────────────────
import numpy as np
import pandas as pd
from datasets import load_dataset, DownloadConfig
import evaluate

# ──────────────────────── local helpers ─────────────────────
from chat_template import prepare_chat_format
from squad_qa_eval import SQuADEvaluator
warnings.filterwarnings("ignore")

# ───────────────────────── logging ──────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Predefined API keys in code
OPENAI_API_KEY = "your_api_key"
ANTHROPIC_API_KEY = "your_api_key"

# ─────────────────────── metrics loaders ────────────────────
metric_bleu = evaluate.load("sacrebleu")
metric_chrf = evaluate.load("chrf")
metric_rouge = evaluate.load("rouge")
metric_exact = evaluate.load("exact_match")
metric_tokens = evaluate.load("seqeval")
metric_squad = SQuADEvaluator()

# ─────────────────── JSON helpers (NumPy safe) ──────────────


def load_sahara_dataset(data_dir, task, cache, attempts=8):
    download_config = DownloadConfig(
        cache_dir=cache,
        max_retries=5,
        resume_download=True,
    )

    for attempt in range(attempts):
        try:
            return load_dataset(
                path=data_dir,
                name=task,
                trust_remote_code=True,
                cache_dir=cache,
                download_config=download_config,
                download_mode="reuse_dataset_if_exists",
            )

        except ConnectionError as exc:
            is_rate_limit = "error 429" in str(exc) or "429" in str(exc)

            if not is_rate_limit or attempt == attempts - 1:
                raise

            delay = min(300, 15 * (2**attempt)) + random.uniform(0, 5)

            logger.warning(
                "Dataset server rate-limited task %s. "
                "Retrying in %.1f seconds (%d/%d)",
                task,
                delay,
                attempt + 1,
                attempts,
            )

            time.sleep(delay)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer, np.int_)):
            return int(o)
        if isinstance(o, (np.floating, np.float_)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


def to_builtin(x):
    if isinstance(x, (np.integer, np.int_)):
        return int(x)
    if isinstance(x, (np.floating, np.float_)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, dict):
        return {k: to_builtin(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_builtin(v) for v in x]
    return x

# ─────────────────── post-processing helpers ─────────────────


def postprocess_text(preds, labels):
    return [p.strip() for p in preds], [[l.strip()] for l in labels]


def correct_mgsm(x: str) -> str:
    if "." in x:
        x = x.rstrip("0").rstrip(".")
    return x.replace(",", "")


def postprocess_mgsm(p): return [correct_mgsm(t.strip()) for t in p]


def postprocess_tokens(preds, labels, pad="O"):
    ps, ls = [], []
    for p, l in zip(preds, labels):
        gold = ast.literal_eval(l.strip())
        pred = p.split()[:len(gold)]+[pad]*max(0, len(gold)-len(p.split()))
        ps.append(pred)
        ls.append(gold)
    return ps, ls


# ─────────────────── unified async LLM client ────────────────
CONCURRENCY = 20


# ─────────────────── inference helpers ───────────────────────
EXAMPLE_SHOWN = False
# Even faster version using translate method


def remove_stop_tokens_fast(text):
    """
    Faster version using str.translate for single character removals
    and replace for multi-character tokens
    """
    # Remove multi-character tokens
    replacements = {
        "<|assistant|>": "",
        "<|user|>": "",
        "<|system|>": "",
        "</s>": "",
        "<|im_end|>": "",
        "<|endoftext|>": "",
        "<end_of_turn>": "",
        "</chat_message>": "",
        "\n\n": ""
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.strip()


async def infer_batch(batch, base, task, max_tok):
    global EXAMPLE_SHOWN
    outs = []
    # for ex in batch:
    for ex in tqdm(batch, desc="Processing Batch"):
        msgs = base.copy()
        input_trim = ex["input"]  # [:int(len(ex["input"])*0.5)]
        if task in ['sentiment', 'topic', 'news', 'xlni', 'mmlu', 'belebele', 'squad_qa', 'lid']:
            msgs.append({"role": "user", "content": input_trim})
        elif task in ['paraphrase', 'title', 'summary']:
            msgs.append(
                {"role": "user", "content": f"language: {ex['lang']}\ntext: {input_trim}"})
        elif task in ['mt_eng2xx', 'mt_fra2xx', 'mt_xx2xx']:
            src, tgt = ex['lang'].split(" to ")
            msgs.append({"role": "user", "content": f"source language: {src}\n"
                         f"target language: {tgt}\n"
                         f"text: {ex['input']}"})
        elif task == "mgsm":
            msgs.append(
                {"role": "user", "content": f"question: {ex['input']}"})
        elif task in ['phrase', 'pos', 'ner']:
            toks = ' '.join(ast.literal_eval(ex['input']))
            msgs.append(
                {"role": "user", "content": f"Language: {ex['lang']}\nText: {toks}"})
        if not EXAMPLE_SHOWN:
            print("==== prompt sample ====")
            for m in msgs:
                print(m)
            EXAMPLE_SHOWN = True
        prompt = prepare_chat_format(msgs, "Sunbird/Sunflower-Qwen3.5-9B")
        outs.append({
            "lang_code": ex["lang_code"],
            "example_id": str(ex.get("id", "")),
            "messages": msgs,
            "max_tokens": max_tok,
            "task": task,
            "example": ex,
            "prompt_with_chat_template": prompt
        })
    return outs

# ─────────────────── benchmark one task ──────────────────────


async def bench_task(provider, task, data_dir, cache, batch):
    global EXAMPLE_SHOWN
    EXAMPLE_SHOWN = False
    t0 = time.time()

    data = data = load_sahara_dataset(
        data_dir=data_dir,
        task=task,
        cache=cache,
    )
    # few-shot setup
    # n_shots=5 if task not in ['topic','news','title','summary', 'lid'] else \
    #       3 if task in ['topic','news'] else 2
    n_shots = 5
    if task in ['topic']:
        n_shots = 3
    elif task in ['lid']:
        n_shots = 10
    elif task in ['title', 'summary', 'news']:
        n_shots = 2

    # choose max tokens per task
    max_tok = {
        'sentiment': 10, 'lid': 10, 'topic': 10, 'news': 10, 'xlni': 10, 'mgsm': 10,
        'title': 50, 'summary': 50,
        'mt_eng2xx': 50, 'mt_fra2xx': 50, 'mt_xx2xx': 50,
        'paraphrase': 50,
        'squad_qa': 50,
        'mmlu': 1, 'belebele': 1,
        'phrase': 100, 'pos': 100, 'ner': 100,
    }.get(task, 1024)

    shots = data['train'].select(range(n_shots))
    choice_from_list = ""
    if task in ['sentiment', 'lid', 'xlni', 'belebele', 'mmlu']:
        choice_from_list = "The answer should be on of the provided list. "
    system_user = "system"
    if provider == "anthropic":
        system_user = "user"
    base = [{"role": system_user,
             "content": f"{shots[0]['instruction']} {choice_from_list}"
            "Return only the bare result; no explanations."}]

    if task in ['sentiment', 'lid', 'topic', 'news', 'xlni',
                'mmlu', 'belebele', 'squad_qa']:
        for shot_example in shots:
            base.extend([
                {"role": "user", "content": shot_example['input']},
                {"role": "assistant", "content": shot_example['output']}
            ])
    elif task in ['paraphrase', 'title', 'summary']:
        for shot_example in shots:
            base.extend([
                {"role": "user",
                    "content": f"language: {shot_example['lang']}\ntext: {shot_example['input']}"},
                {"role": "assistant", "content": shot_example['output']}
            ])

    elif task in ['mt_eng2xx', 'mt_fra2xx', 'mt_xx2xx']:
        for shot_example in shots:
            langs_info = shot_example['lang'].split(" to ")
            source_lang = langs_info[0]
            target_lang = langs_info[1]
            base.extend([
                {"role": "user",
                    "content": f"source language: {source_lang}\ntarget language: {target_lang}\ntext: {shot_example['input']}"},
                {"role": "assistant", "content": shot_example['output']}
            ])

    elif task in ['mgsm']:
        for shot_example in shots:
            base.extend([
                {"role": "user",
                    "content": f"question: {shot_example['input']}"},
                {"role": "assistant", "content": shot_example['output']}
            ])
    elif task in ['phrase', 'pos', 'ner']:
        for shot_example in shots:
            base.extend([
                {"role": "user",
                    "content": f"Language: {shot_example['lang']}\nText: {' '.join(ast.literal_eval(shot_example['input']))}"},
                {"role": "assistant",
                    "content": f"{' '.join(ast.literal_eval(shot_example['output']))}"}
            ])

    test = list(data['test'])
    results = []
    for i in range(0, len(test), batch):
        results.extend(await infer_batch(test[i:i+batch], base, task, max_tok))
        logger.info("%s batch %d/%d", task,
                    i//batch+1, (len(test)+batch-1)//batch)
        if provider not in ["vllm", "vllm_server", "transformers"]:
            time.sleep(10)
        # break
    df = pd.DataFrame(results)
    return df, round(time.time()-t0, 2)

# ─────────────────── orchestrator ────────────────────────────


async def main_async(provider, model_id, tasks, data_dir, cache, batch):
    # Change this line to await the new create method

    model_safe = model_id.replace("/", "_").replace("-", "_")
    out_dir = f"outputs/{model_safe}"
    os.makedirs(out_dir, exist_ok=True)
    for t in tasks:
        df, sec = await bench_task(provider, t, data_dir, cache, batch)
        df.to_json(f"{out_dir}/{t}_generation.json",
                   orient="records", force_ascii=False)
        logger.info("%s finished (%.1fs): %s", t, sec, "")
# ─────────────────── CLI ─────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    # Add "transformers" to the list of choices
    p.add_argument("--provider", choices=["openai", "anthropic",
                   "vllm", "vllm_server", "transformers"], required=True)
    p.add_argument("--model_id", required=True,
                   help="For Claude 4, use: claude-sonnet-4-20250514 or claude-opus-4-20250514")
    p.add_argument("--tasks", nargs="+", default=["sentiment"])
    p.add_argument("--sahara_dir",
                   default="UBC-NLP/sahara_benchmark")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


def main():
    a = parse_args()
    print("==========================================================")
    print("==========================================================")
    print("==========================================================")
    print("Running Sahara-v1 benchmark for collecting model prompts")
    print("===========================================================")
    print("===========================================================")
    print("===========================================================")
    logger.info("provider=%s model=%s tasks=%s",
                a.provider, a.model_id, a.tasks)
    asyncio.run(main_async(a.provider, a.model_id, a.tasks,
                           a.sahara_dir, a.cache_dir, a.batch_size))


if __name__ == "__main__":
    main()
