
#!/usr/bin/env python3
"""
Audio Model Evaluation Module (optimized + feature cache + per-sample)
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
import os, json, hashlib

import numpy as np
import soundfile as sf
import librosa

try:
    from pystoi import stoi as _stoi
    HAS_STOI = True
except Exception:
    HAS_STOI = False

TARGET_SR = 16000

def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

@dataclass
class AudioEvalResult:
    snr_mean: float
    snr_std: float
    stoi_mean: float
    stoi_std: float
    per_sample: Optional[List[Dict[str, Any]]] = None

def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)

def _load_wav(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    y, sr = sf.read(path, always_2d=False)
    if y.dtype != np.float32 and y.dtype != np.float64:
        y = y.astype(np.float32) / np.iinfo(y.dtype).max
    y = _to_mono(y)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y.astype(np.float32)

def _align(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    if n <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    return a[:n], b[:n]

def _snr(ref: np.ndarray, est: np.ndarray, eps: float = 1e-12) -> float:
    ref, est = _align(ref, est)
    if len(ref) == 0:
        return 0.0
    noise = ref - est
    p_sig = np.mean(ref ** 2)
    p_noise = np.mean(noise ** 2) + eps
    if p_sig <= eps:
        return 0.0
    return 10.0 * np.log10(p_sig / p_noise)

def _stoi_wrap(ref: np.ndarray, est: np.ndarray, sr: int = TARGET_SR) -> float:
    ref, est = _align(ref, est)
    if len(ref) == 0 or not HAS_STOI:
        return 0.0
    try:
        return float(_stoi(ref, est, sr, extended=False))
    except Exception:
        return 0.0

def _mfcc(y: np.ndarray, sr: int = TARGET_SR, n_mfcc: int = 20) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    logS = librosa.power_to_db(S, ref=np.max)
    M = librosa.feature.mfcc(S=librosa.db_to_amplitude(logS), sr=sr, n_mfcc=n_mfcc)
    # Aggregate over time by mean to a fixed vector
    return M.mean(axis=1).astype(np.float32)

class AudioEvaluator:
    def __init__(self, target_sr: int = TARGET_SR, cache_dir: Optional[str] = None):
        self.sr = target_sr
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.feat_dir = os.path.join(self.cache_dir, "audio_feats")
            os.makedirs(self.feat_dir, exist_ok=True)
        else:
            self.feat_dir = None

    def _features_with_cache(self, path: str, y: np.ndarray) -> np.ndarray:
        if self.feat_dir and path:
            key = _sha(path) + ".npy"
            fpath = os.path.join(self.feat_dir, key)
            if os.path.exists(fpath):
                try:
                    return np.load(fpath)
                except Exception:
                    pass
            feat = _mfcc(y, self.sr)
            np.save(fpath, feat)
            return feat
        return _mfcc(y, self.sr)

    def evaluate_pairs(self, reference_paths: List[str], candidate_paths: List[str], per_sample: bool = False) -> AudioEvalResult:
        n = min(len(reference_paths), len(candidate_paths))
        snrs, stois = [], []
        per = [] if per_sample else None
        for r, c in zip(reference_paths[:n], candidate_paths[:n]):
            ref = _load_wav(r, self.sr)
            est = _load_wav(c, self.sr)

            sn = _snr(ref, est)
            snrs.append(sn)
            st = _stoi_wrap(ref, est, self.sr)
            stois.append(st)

            if per_sample:
                # Cache features (MFCC) for analysis
                f_ref = self._features_with_cache(r, ref)
                f_est = self._features_with_cache(c, est)
                per.append({
                    "ref": r,
                    "cand": c,
                    "snr": float(sn),
                    "stoi": float(st),
                    "ref_mfcc": f_ref.tolist(),
                    "cand_mfcc": f_est.tolist(),
                })

        snr_mean = float(np.mean(snrs)) if snrs else 0.0
        snr_std = float(np.std(snrs)) if snrs else 0.0
        stoi_mean = float(np.mean(stois)) if stois else 0.0
        stoi_std = float(np.std(stois)) if stois else 0.0
        return AudioEvalResult(snr_mean, snr_std, stoi_mean, stoi_std, per_sample=per)

    def evaluate_audio_model(self, reference_audios: List[np.ndarray], generated_audios: List[np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate an audio generation model using multiple metrics.
        
        Args:
            reference_audios (List[np.ndarray]): List of reference audio signals
            generated_audios (List[np.ndarray]): List of generated audio signals
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        try:
            n = min(len(reference_audios), len(generated_audios))
            snrs, stois = [], []
            
            for i in range(n):
                ref = reference_audios[i]
                gen = generated_audios[i]
                
                # Calculate SNR
                sn = _snr(ref, gen)
                snrs.append(sn)
                
                # Calculate STOI
                st = _stoi_wrap(ref, gen, self.sr)
                stois.append(st)
            
            snr_mean = float(np.mean(snrs)) if snrs else 0.0
            snr_std = float(np.std(snrs)) if snrs else 0.0
            stoi_mean = float(np.mean(stois)) if stois else 0.0
            stoi_std = float(np.std(stois)) if stois else 0.0
            
            # Calculate overall score (higher is better for both SNR and STOI)
            normalized_snr = min(1, max(0, (snr_mean + 10) / 40))  # Normalize -10 to 30 dB range
            normalized_stoi = stoi_mean  # STOI is already 0-1
            
            overall_score = (normalized_snr + normalized_stoi) / 2
            
            return {
                'snr': {
                    'mean': snr_mean,
                    'std': snr_std
                },
                'stoi': {
                    'mean': stoi_mean,
                    'std': stoi_std
                },
                'overall_score': overall_score
            }
            
        except Exception as e:
            print(f"Error evaluating audio model: {e}")
            return {
                'snr': {'mean': 0.0, 'std': 0.0},
                'stoi': {'mean': 0.0, 'std': 0.0},
                'overall_score': 0.0
            }

    def generate_report(self, results: Dict[str, Any]) -> str:
        report = "=== AUDIO MODEL EVALUATION REPORT ===\n"
        report += f"\nOverall Score: {results['overall_score']:.4f}\n"
        report += f"\nSTOI Scores:\n"
        report += f"  Mean: {results['stoi']['mean']:.4f}\n"
        report += f"  Std: {results['stoi']['std']:.4f}\n"
        report += f"\nSNR Scores:\n"
        report += f"  Mean: {results['snr']['mean']:.4f} dB\n"
        report += f"  Std: {results['snr']['std']:.4f} dB\n"
        return report

    def load_audio(self, filepath: str) -> np.ndarray:
        audio, sr = sf.read(filepath)
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
        return audio

# --- CLI ---
def _cli():
    import argparse, glob, sys, json

    p = argparse.ArgumentParser(description="Audio evaluation for paired files")
    p.add_argument("--refs", type=str, help="Glob for reference wavs (e.g., data/refs/*.wav)")
    p.add_argument("--cands", type=str, help="Glob for candidate wavs (e.g., data/cands/*.wav)")
    p.add_argument("--cache_dir", type=str, default=None, help="Cache dir for audio features")
    p.add_argument("--per_sample", action="store_true", help="Return per-sample metrics and features")
    args = p.parse_args()

    if not args.refs or not args.cands:
        print("Provide --refs and --cands globs")
        sys.exit(2)

    ref_paths = sorted(glob.glob(args.refs))
    cand_paths = sorted(glob.glob(args.cands))
    if not ref_paths or not cand_paths:
        print("No files matched globs")
        sys.exit(2)

    ev = AudioEvaluator(cache_dir=args.cache_dir)
    res = ev.evaluate_pairs(ref_paths, cand_paths, per_sample=args.per_sample)
    print(json.dumps(asdict(res), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    _cli()
