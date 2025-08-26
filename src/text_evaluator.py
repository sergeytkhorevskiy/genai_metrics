#!/usr/bin/env python3

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import math, json, hashlib, os

import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import sacrebleu

_HAS_COMET = False
try:
    from comet import download_model, load_from_checkpoint
    _HAS_COMET = True
except Exception:
    _HAS_COMET = False

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception:
        pass

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class TextEvalResult:
    bleu: float
    sacrebleu: float
    chrf: float
    rougeL_f: float
    rouge1_f: float
    rouge2_f: float
    length_ratio_mean: float
    length_ratio_std: float
    overall_score: float
    per_sample: Optional[List[Dict[str, Any]]] = None


class TextEvaluator:
    def __init__(self, cache_dir: Optional[str] = None, use_comet: bool = False):
        self._tok_mem: Dict[str, List[str]] = {}
        self.cache_dir = cache_dir
        self.use_comet = use_comet and _HAS_COMET
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.tok_dir = os.path.join(self.cache_dir, "text_toks")
            os.makedirs(self.tok_dir, exist_ok=True)
        else:
            self.tok_dir = None

        self._comet_model = None
        if self.use_comet:
            try:
                ckpt_path = download_model("Unbabel/wmt22-comet-da")
                self._comet_model = load_from_checkpoint(ckpt_path)
            except Exception:
                self._comet_model = None

    def _tok(self, s: str) -> List[str]:
        if s in self._tok_mem:
            return self._tok_mem[s]
        if self.tok_dir:
            key = _sha(s) + ".json"
            fpath = os.path.join(self.tok_dir, key)
            if os.path.exists(fpath):
                try:
                    toks = json.loads(open(fpath, "r", encoding="utf-8").read())
                    self._tok_mem[s] = toks
                    return toks
                except Exception:
                    pass
        toks = word_tokenize(s) if s else []
        self._tok_mem[s] = toks
        if self.tok_dir:
            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(toks, f, ensure_ascii=False)
            except Exception:
                pass
        return toks

    def _normalize_lists(self, references: List[str], candidates: List[str]) -> Tuple[List[str], List[str]]:
        refs = [r if isinstance(r, str) else "" for r in references]
        cands = [c if isinstance(c, str) else "" for c in candidates]
        n = min(len(refs), len(cands))
        return refs[:n], cands[:n]

    def _safe_len_ratio_stats(self, references: List[str], candidates: List[str]) -> Tuple[float, float, List[float]]:
        ratios = []
        for r, c in zip(references, candidates):
            lr = len(c.split()) / max(1, len(r.split()))
            ratios.append(lr)
        if not ratios:
            return 0.0, 0.0, []
        arr = np.array(ratios, dtype=np.float64)
        return float(arr.mean()), float(arr.std()), ratios

    def _compute_bleu(self, references: List[str], candidates: List[str]) -> float:
        if not references or not candidates:
            return 0.0
        refs_tok = [[self._tok(r)] for r in references]  # list of list-of-refs
        cands_tok = [self._tok(c) for c in candidates]
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        chencherry = SmoothingFunction()
        score = corpus_bleu(refs_tok, cands_tok, smoothing_function=chencherry.method3)
        return float(score)

    def _compute_sacrebleu(self, references: List[str], candidates: List[str]) -> float:
        if not references or not candidates:
            return 0.0
        score = sacrebleu.corpus_bleu(candidates, [references]).score
        return float(score / 100.0)

    def _compute_chrf(self, references: List[str], candidates: List[str]) -> float:
        if not references or not candidates:
            return 0.0
        score = sacrebleu.corpus_chrf(candidates, [references]).score
        return float(score / 100.0)

    def _compute_rouge(self, references: List[str], candidates: List[str]) -> Tuple[float, float, float, List[float]]:
        if not references or not candidates:
            return 0.0, 0.0, 0.0, []
        scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeLsum'], use_stemmer=True)
        r1, r2, rL = [], [], []
        per_rl = []
        for r, c in zip(references, candidates):
            scores = scorer.score(r, c)
            r1.append(scores['rouge1'].fmeasure)
            r2.append(scores['rouge2'].fmeasure)
            rL.append(scores['rougeLsum'].fmeasure)
            per_rl.append(scores['rougeLsum'].fmeasure)
        r1m = float(np.mean(r1)) if r1 else 0.0
        r2m = float(np.mean(r2)) if r2 else 0.0
        rLm = float(np.mean(rL)) if rL else 0.0
        return rLm, r1m, r2m, per_rl

    def _sentence_bleu_list(self, references: List[str], candidates: List[str]) -> List[float]:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ch = SmoothingFunction()
        scores = []
        for r, c in zip(references, candidates):
            ref_tok = self._tok(r)
            cand_tok = self._tok(c)
            if not cand_tok:
                scores.append(0.0)
            else:
                try:
                    scores.append(float(sentence_bleu([ref_tok], cand_tok, smoothing_function=ch.method3)))
                except Exception:
                    scores.append(0.0)
        return scores

    def _comet_scores(self, references: List[str], candidates: List[str], sources: Optional[List[str]] = None) -> Optional[List[float]]:
        if not self._comet_model:
            return None
        data = []
        for r, c in zip(references, candidates):
            data.append({"src": sources[0] if sources else "", "mt": c, "ref": r})
        try:
            seg_scores, sys_score = self._comet_model.predict(data, batch_size=8, gpus=0)
            return [float(s) for s in seg_scores]
        except Exception:
            return None

    def evaluate(self, references: List[str], candidates: List[str], per_sample: bool = False) -> TextEvalResult:
        references, candidates = self._normalize_lists(references, candidates)

        bleu = self._compute_bleu(references, candidates)
        sbleu = self._compute_sacrebleu(references, candidates)
        chrf = self._compute_chrf(references, candidates)
        rougeL_f, rouge1_f, rouge2_f, per_rl = self._compute_rouge(references, candidates)
        lr_mean, lr_std, lr_list = self._safe_len_ratio_stats(references, candidates)

        comps = [bleu, sbleu, chrf, rouge1_f, rouge2_f, rougeL_f]
        overall = float(np.mean(comps)) if comps else 0.0

        per = None
        if per_sample:
            sent_bleu = self._sentence_bleu_list(references, candidates)
            comet = self._comet_scores(references, candidates) if self.use_comet else None
            per = []
            for i, (r, c) in enumerate(zip(references, candidates)):
                d = {
                    "idx": i,
                    "reference": r,
                    "candidate": c,
                    "sentence_bleu": float(sent_bleu[i]),
                    "rougeL_f": float(per_rl[i] if i < len(per_rl) else 0.0),
                    "len_ratio": float(lr_list[i] if i < len(lr_list) else 0.0),
                }
                if comet is not None:
                    d["comet"] = float(comet[i])
                per.append(d)

        return TextEvalResult(
            bleu=bleu,
            sacrebleu=sbleu,
            chrf=chrf,
            rougeL_f=rougeL_f,
            rouge1_f=rouge1_f,
            rouge2_f=rouge2_f,
            length_ratio_mean=lr_mean,
            length_ratio_std=lr_std,
            overall_score=overall,
            per_sample=per
        )

    def evaluate_text_model(self, references: List[str], candidates: List[str]) -> Dict[str, Any]:
        result = self.evaluate(references, candidates)
        
        return {
            'overall_score': result.overall_score,
            'bleu': {
                'bleu_score': result.bleu,
                'sacrebleu_score': result.sacrebleu,
                'chrf_score': result.chrf
            },
            'rouge': {
                'rouge1': {'fmeasure': result.rouge1_f},
                'rouge2': {'fmeasure': result.rouge2_f},
                'rougeL': {'fmeasure': result.rougeL_f}
            },
            'length_metrics': {
                'avg_reference_length': 0.0,
                'avg_candidate_length': 0.0,
                'avg_length_ratio': result.length_ratio_mean,
                'length_ratio_std': result.length_ratio_std
            }
        }

    def generate_report(self, results: Dict[str, Any]) -> str:
        report = "=== TEXT MODEL EVALUATION REPORT ===\n"
        report += f"\nOverall Score: {results['overall_score']:.4f}\n"
        report += f"\nBLEU Scores:\n"
        report += f"  BLEU Score: {results['bleu']['bleu_score']:.4f}\n"
        report += f"  SacreBLEU Score: {results['bleu']['sacrebleu_score']:.4f}\n"
        report += f"  chrF Score: {results['bleu']['chrf_score']:.4f}\n"
        report += f"\nROUGE Scores:\n"
        report += f"  ROUGE1 F1: {results['rouge']['rouge1']['fmeasure']:.4f}\n"
        report += f"  ROUGE2 F1: {results['rouge']['rouge2']['fmeasure']:.4f}\n"
        report += f"  ROUGEL F1: {results['rouge']['rougeL']['fmeasure']:.4f}\n"
        report += f"\nLength Metrics:\n"
        report += f"  Length Ratio Mean: {results['length_metrics']['avg_length_ratio']:.4f}\n"
        report += f"  Length Ratio Std: {results['length_metrics']['length_ratio_std']:.4f}\n"
        return report

# --- CLI ---
def _cli():
    import argparse, sys, pathlib

    p = argparse.ArgumentParser(description="Text evaluation")
    p.add_argument("--refs", type=str, help="Path to references TXT/JSON")
    p.add_argument("--cands", type=str, help="Path to candidates TXT/JSON")
    p.add_argument("--cache_dir", type=str, default=None, help="Cache dir for tokenization")
    p.add_argument("--use_comet", action="store_true", help="Enable COMET segment scores (optional)")
    p.add_argument("--per_sample", action="store_true", help="Return per-sample metrics")
    args = p.parse_args()

    def _load(path: str) -> list:
        if path.lower().endswith(".json"):
            return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        return [line.rstrip("\n") for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()]

    if not args.refs or not args.cands:
        print("Provide --refs and --cands")
        sys.exit(2)

    refs = _load(args.refs)
    cands = _load(args.cands)

    evaluator = TextEvaluator(cache_dir=args.cache_dir, use_comet=args.use_comet)
    res = evaluator.evaluate(refs, cands, per_sample=args.per_sample)
    print(json.dumps(asdict(res), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    _cli()
