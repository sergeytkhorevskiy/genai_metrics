import os
import math
import glob
import logging
from typing import List, Optional, Tuple, Dict, Any
import hashlib

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms as T

logger = logging.getLogger("image_evaluator")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

def _pil_loader(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")

def load_images_from_dir(directory: str, limit: Optional[int] = None) -> List[Image.Image]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(directory, p)))
    files = sorted(files)
    if limit is not None:
        files = files[:limit]
    images = []
    for f in files:
        try:
            images.append(_pil_loader(f))
        except Exception as e:
            logger.warning(f"Failed to load image '{f}': {e}")
    return images

def _numpy_mean_cov(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if feats.ndim != 2:
        feats = feats.reshape(feats.shape[0], -1)
    feats = feats.astype(np.float64)
    mu = np.mean(feats, axis=0)
    
    n_samples = feats.shape[0]
    if n_samples > 1:
        centered = feats - mu
        sigma = (centered.T @ centered) / (n_samples - 1)
    else:
        sigma = np.eye(feats.shape[1])
    
    return mu, sigma

def _covmean_sqrt(sigma1: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    eps = 1e-6
    try:
        from scipy import linalg
        covmean = linalg.sqrtm((sigma1 + eps * np.eye(sigma1.shape[0])) @
                               (sigma2 + eps * np.eye(sigma2.shape[0])))
        covmean = np.real_if_close(covmean, tol=1000)
        if np.isfinite(covmean).all():
            return covmean
    except Exception:
        pass

    prod = (sigma1 @ sigma2 + sigma2 @ sigma1) / 2.0
    prod = (prod + prod.T) / 2.0
    U, S, Vt = np.linalg.svd(prod, full_matrices=False)
    S = np.clip(S, 0.0, None)
    covmean = (U * np.sqrt(S)) @ Vt
    return covmean

def _hash_file(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()[:16]

def _pairwise_distances_flat(feats: np.ndarray) -> np.ndarray:
    n = feats.shape[0]
    if n < 2:
        return np.array([])
    
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(feats[i] - feats[j])
            dists.append(dist)
    return np.array(dists)


def _pairwise_distances_flat(features: np.ndarray) -> np.ndarray:
    """Return the upper-triangular (k=1) distances flattened as 1D array."""
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
    X = features.astype(np.float64)
    G = X @ X.T
    sq = np.diag(G)
    D2 = sq[:, None] + sq[None, :] - 2.0 * G
    D2 = np.maximum(D2, 0.0)
    D = np.sqrt(D2)
    iu = np.triu_indices_from(D, k=1)
    return D[iu]


# ----------------------------
# Inception Wrappers
# ----------------------------



def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class InceptionFeatureExtractor(nn.Module):
    """InceptionV3 feature extractor for FID/diversity (2048-d)."""

    def __init__(self, device: torch.device):
        super().__init__()
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        model = models.inception_v3(weights=weights, aux_logits=True)
        model.fc = nn.Identity()
        model.eval()
        self.model = model.to(device)
        self.transform = weights.transforms()

        self.device = device

    @torch.no_grad()
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # batch: [N, 3, H, W], already transformed to [299, 299]
        feats = self.model(batch)  # [N, 2048]
        return feats

    def extract_features_batched(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        feats = []
        for i in range(0, len(images), batch_size):
            chunk = images[i:i + batch_size]
            t = torch.stack([self.transform(img) for img in chunk], dim=0).to(self.device)
            f = self.forward(t)  # [B, 2048]
            feats.append(f.cpu().numpy())
        if not feats:
            return np.empty((0, 2048), dtype=np.float32)
        return np.concatenate(feats, axis=0)


class InceptionClassifier(nn.Module):
    """InceptionV3 logits for Inception Score (1000-way softmax)."""

    def __init__(self, device: torch.device):
        super().__init__()
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        model = models.inception_v3(weights=weights, aux_logits=True)
        model.eval()
        self.model = model.to(device)
        self.transform = weights.transforms()
        self.device = device

    @torch.no_grad()
    def predict(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        preds = []
        for i in range(0, len(images), batch_size):
            chunk = images[i:i + batch_size]
            t = torch.stack([self.transform(img) for img in chunk], dim=0).to(self.device)
            logits = self.model(t)  # [B, 1000]
            prob = F.softmax(logits, dim=1).cpu().numpy()
            preds.append(prob)
        if not preds:
            return np.empty((0, 1000), dtype=np.float32)
        return np.concatenate(preds, axis=0)


# ----------------------------
# Evaluator
# ----------------------------

class ImageEvaluator:
    def __init__(self, device: Optional[str] = None, cache_dir: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")

        self._feat_extractor = InceptionFeatureExtractor(self.device)
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self._classifier = InceptionClassifier(self.device)

    def save_features(self, images: List[Image.Image], path: str, batch_size: int = 32):
        feats = self._feat_extractor.extract_features_batched(images, batch_size=batch_size)
        import numpy as np, pathlib
        pathlib.Path(path).write_bytes(feats.astype(np.float32).tobytes())

    def load_features(self, path: str, dim: int = 2048) -> np.ndarray:
        import numpy as np, pathlib
        arr = np.frombuffer(pathlib.Path(path).read_bytes(), dtype=np.float32)
        return arr.reshape(-1, dim)

    def _features_with_cache(self, images: List[Image.Image], image_paths: Optional[List[str]] = None, batch_size: int = 32) -> np.ndarray:
        if self.cache_dir and image_paths and len(images) == len(image_paths):
            feats_list = []
            to_compute_imgs, to_compute_idx = [], []
            for i, (img, p) in enumerate(zip(images, image_paths)):
                key = _hash_file(p) + ".npy"
                cpath = os.path.join(self.cache_dir, key)
                if os.path.exists(cpath):
                    try:
                        feats_list.append(np.load(cpath))
                        continue
                    except Exception:
                        pass
                to_compute_imgs.append(img)
                to_compute_idx.append((i, cpath))
                feats_list.append(None)

            if to_compute_imgs:
                batch_feats = self._feat_extractor.extract_features_batched(to_compute_imgs, batch_size=batch_size)
                j = 0
                for i, cpath in to_compute_idx:
                    f = batch_feats[j]
                    np.save(cpath, f)
                    feats_list[i] = f
                    j += 1
            return np.stack(feats_list, axis=0).astype(np.float64)

        return self._feat_extractor.extract_features_batched(images, batch_size=batch_size).astype(np.float64)

# --- FID ---

    def calculate_fid(
        self,
        real_images: List[Image.Image],
        generated_images: List[Image.Image],
        batch_size: int = 32,
        max_images: Optional[int] = None,
    ) -> float:
        """Compute FID between real_images and generated_images (lists of PIL)."""
        if max_images is not None:
            real_images = real_images[:max_images]
            generated_images = generated_images[:max_images]

        if len(real_images) == 0 or len(generated_images) == 0:
            logger.warning("Empty image lists provided to FID.")
            return float("inf")

        try:
            real_paths = [getattr(img, 'filename', None) for img in real_images]
            gen_paths = [getattr(img, 'filename', None) for img in generated_images]
            real_feats = self._features_with_cache(real_images, real_paths, batch_size=batch_size)
            gen_feats = self._features_with_cache(generated_images, gen_paths, batch_size=batch_size)

            mu_r, sig_r = _numpy_mean_cov(real_feats)
            mu_g, sig_g = _numpy_mean_cov(gen_feats)

            diff = mu_r - mu_g
            covmean = _covmean_sqrt(sig_r, sig_g)
            if not np.isfinite(covmean).all():
                logger.warning("Non-finite covmean encountered; returning inf FID.")
                return float("inf")
            fid = float(diff @ diff + np.trace(sig_r + sig_g - 2.0 * covmean))
            return fid
        except Exception as e:
            logger.error(f"Error while computing FID: {e}")
            return float("inf")

    # --- Inception Score ---
    def calculate_inception_score(
        self,
        images: List[Image.Image],
        splits: int = 5,
        batch_size: int = 32,
        max_images: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Compute Inception Score (mean, std)."""
        if max_images is not None:
            images = images[:max_images]

        n = len(images)
        if n == 0:
            logger.warning("Empty image list provided to Inception Score.")
            return 0.0, 0.0

        splits = max(1, min(splits, n))

        try:
            preds = self._classifier.predict(images, batch_size=batch_size)  # [N, 1000]
            preds = np.clip(preds, 1e-16, 1.0)

            split_size = n // splits
            scores = []
            for i in range(splits):
                s = i * split_size
                e = n if i == splits - 1 else s + split_size
                split = preds[s:e]
                if len(split) == 0:
                    continue
                p_y = np.mean(split, axis=0, keepdims=True)
                kl = split * (np.log(split) - np.log(p_y))
                score = float(np.exp(np.mean(np.sum(kl, axis=1))))
                scores.append(score)

            if not scores:
                return 0.0, 0.0
            return float(np.mean(scores)), float(np.std(scores))
        except Exception as e:
            logger.error(f"Error while computing Inception Score: {e}")
            return 0.0, 0.0

    # --- Diversity ---
    def calculate_diversity_metrics(
        self,
        images: List[Image.Image],
        batch_size: int = 32,
        max_images: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Return (mean_pairwise_distance, median_pairwise_distance) in feature space."""
        if max_images is not None:
            images = images[:max_images]
        if len(images) < 2:
            logger.warning("Need at least 2 images for diversity metrics.")
            return 0.0, 0.0

        try:
            paths = [getattr(img, 'filename', None) for img in images]
            feats = self._features_with_cache(images, paths, batch_size=batch_size)  # [N, 2048]
            dists = _pairwise_distances_flat(feats)  # [N*(N-1)/2]
            return float(np.mean(dists)), float(np.median(dists))
        except Exception as e:
            logger.error(f"Error while computing diversity metrics: {e}")
            return 0.0, 0.0

    def evaluate_image_model(self, real_images: List[Image.Image], generated_images: List[Image.Image]) -> Dict[str, Any]:
        try:
            fid_score = self.calculate_fid(real_images, generated_images, max_images=10)
            
            inception_mean, inception_std = self.calculate_inception_score(generated_images, max_images=10)
            
            diversity_mean, diversity_median = self.calculate_diversity_metrics(generated_images, max_images=10)
            
            normalized_fid = max(0, 1 - (fid_score / 500))
            normalized_inception = min(1, inception_mean / 10)
            
            overall_score = (normalized_fid + normalized_inception) / 2
            
            return {
                'fid_score': fid_score,
                'inception_score': {
                    'mean': inception_mean,
                    'std': inception_std
                },
                'diversity_metrics': {
                    'mean_pairwise_distance': diversity_mean,
                    'median_pairwise_distance': diversity_median
                },
                'overall_score': overall_score
            }
            
        except Exception as e:
            logger.error(f"Error evaluating image model: {e}")
            return {
                'fid_score': float('inf'),
                'inception_score': {'mean': 0.0, 'std': 0.0},
                'diversity_metrics': {'mean_pairwise_distance': 0.0, 'median_pairwise_distance': 0.0},
                'overall_score': 0.0
            }

    def generate_report(self, results: Dict[str, Any]) -> str:
        report = "=== IMAGE MODEL EVALUATION REPORT ===\n"
        report += f"\nOverall Score: {results['overall_score']:.4f}\n"
        report += f"\nFID Score:\n"
        report += f"  FID: {results['fid_score']:.4f}\n"
        report += f"  (Lower is better, typical range: 0-500)\n"
        report += f"\nInception Score:\n"
        report += f"  Mean: {results['inception_score']['mean']:.4f}\n"
        report += f"  Std: {results['inception_score']['std']:.4f}\n"
        report += f"  (Higher is better, typical range: 1-10)\n"
        report += f"\nDiversity Metrics:\n"
        report += f"  Mean Pairwise Distance: {results['diversity_metrics']['mean_pairwise_distance']:.4f}\n"
        report += f"  Median Pairwise Distance: {results['diversity_metrics']['median_pairwise_distance']:.4f}\n"
        return report


# ----------------------------
# CLI (optional)
# ----------------------------

def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate image sets with FID/IS/Diversity.")
    parser.add_argument("--real_dir", type=str, help="Directory with real images")
    parser.add_argument("--gen_dir", type=str, help="Directory with generated images")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--max_images", type=int, default=None, help="Optional cap on number of images")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--splits", type=int, default=5)

    args = parser.parse_args()

    evaluator = ImageEvaluator(device=args.device)

    results = {}

    if args.real_dir and args.gen_dir:
        real_imgs = load_images_from_dir(args.real_dir, limit=args.max_images)
        gen_imgs = load_images_from_dir(args.gen_dir, limit=args.max_images)
        fid = evaluator.calculate_fid(real_imgs, gen_imgs, batch_size=args.batch_size, max_images=args.max_images)
        results["FID"] = fid
        logger.info(f"FID: {fid:.4f}")

    # Inception Score on generated images if provided
    if args.gen_dir:
        gen_imgs = load_images_from_dir(args.gen_dir, limit=args.max_images)
        is_mean, is_std = evaluator.calculate_inception_score(
            gen_imgs, splits=args.splits, batch_size=args.batch_size, max_images=args.max_images
        )
        results["IS_mean"] = is_mean
        results["IS_std"] = is_std
        logger.info(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")

        div_mean, div_median = evaluator.calculate_diversity_metrics(
            gen_imgs, batch_size=args.batch_size, max_images=args.max_images
        )
        results["Diversity_mean_dist"] = div_mean
        results["Diversity_median_dist"] = div_median
        logger.info(f"Diversity (mean, median) pairwise dist: {div_mean:.4f}, {div_median:.4f}")

    if not results:
        logger.info("Nothing to compute. Provide --real_dir and/or --gen_dir.")

if __name__ == "__main__":
    _cli()
