import os
import json
import pandas as pd
from typing import Dict, List, Union, Optional, Any
from datetime import datetime

from .text_evaluator import TextEvaluator
from .image_evaluator import ImageEvaluator
from .audio_evaluator import AudioEvaluator

class GenAIModelEvaluator:
    
    def __init__(self, device: str = 'auto', sample_rate: int = 16000, 
                 resource_mode: str = 'balanced'):
        if device == 'auto':
            try:
                import torch
                if torch.cuda.is_available():
                    device = 'cuda'
                    print(f"✅ Using CUDA device: {torch.cuda.get_device_name(0)}")
                else:
                    device = 'cpu'
                    print("ℹ️  CUDA not available, using CPU")
            except ImportError:
                device = 'cpu'
                print("ℹ️  PyTorch not available, using CPU")
        
        elif device == 'cuda':
            try:
                import torch
                if not torch.cuda.is_available():
                    print("Warning: CUDA requested but not available. Falling back to CPU.")
                    device = 'cpu'
                else:
                    print(f"✅ Using CUDA device: {torch.cuda.get_device_name(0)}")
            except ImportError:
                print("Warning: PyTorch not available. Falling back to CPU.")
                device = 'cpu'
        
        self.device = device
        self.sample_rate = sample_rate
        self.resource_mode = resource_mode
        
        if resource_mode == 'lightweight':
            self.max_images = 5
            self.max_audio_samples = 3
            self.batch_size = 1
            print("🔧 Resource mode: Lightweight (minimal memory usage)")
        elif resource_mode == 'balanced':
            self.max_images = 10
            self.max_audio_samples = 5
            self.batch_size = 2
            print("🔧 Resource mode: Balanced (moderate memory usage)")
        else:
            self.max_images = 50
            self.max_audio_samples = 10
            self.batch_size = 5
            print("🔧 Resource mode: Full (maximum accuracy)")
        
        self.text_evaluator = TextEvaluator()
        self.image_evaluator = ImageEvaluator(device=device)
        self.audio_evaluator = AudioEvaluator(target_sr=sample_rate)
        
        self.evaluation_results = {}
    
    def evaluate_text_model(self, references: List[str], candidates: List[str], 
                          model_name: str = "text_model") -> Dict[str, Any]:
        print(f"Evaluating text model: {model_name}")
        
        try:
            results = self.text_evaluator.evaluate_text_model(references, candidates)
            results['model_name'] = model_name
            results['model_type'] = 'text'
            results['timestamp'] = datetime.now().isoformat()
            
            self.evaluation_results[model_name] = results
            
            report = self.text_evaluator.generate_report(results)
            print(report)
            
            return results
            
        except Exception as e:
            print(f"Error evaluating text model: {e}")
            return {}
    
    def evaluate_image_model(self, real_images: List, generated_images: List, 
                           model_name: str = "image_model") -> Dict[str, Any]:
        print(f"Evaluating image model: {model_name}")
        
        try:
            from PIL import Image
            
            real_pil_images = []
            for img in real_images:
                if isinstance(img, str):
                    real_pil_images.append(Image.open(img).convert('RGB'))
                else:
                    real_pil_images.append(img)
            
            generated_pil_images = []
            for img in generated_images:
                if isinstance(img, str):
                    generated_pil_images.append(Image.open(img).convert('RGB'))
                else:
                    generated_pil_images.append(img)
            
            if len(real_pil_images) > self.max_images or len(generated_pil_images) > self.max_images:
                print(f"⚠️  Limiting image evaluation to {self.max_images} images due to resource mode '{self.resource_mode}'")
                real_pil_images = real_pil_images[:self.max_images]
                generated_pil_images = generated_pil_images[:self.max_images]
            
            results = self.image_evaluator.evaluate_image_model(real_pil_images, generated_pil_images)
            results['model_name'] = model_name
            results['model_type'] = 'image'
            results['timestamp'] = datetime.now().isoformat()
            
            self.evaluation_results[model_name] = results
            
            report = self.image_evaluator.generate_report(results)
            print(report)
            
            return results
            
        except Exception as e:
            print(f"Error evaluating image model: {e}")
            return {}
    
    def evaluate_audio_model(self, reference_audios: List, generated_audios: List, 
                           model_name: str = "audio_model") -> Dict[str, Any]:
        print(f"Evaluating audio model: {model_name}")
        
        try:
            import numpy as np
            
            reference_signals = []
            for audio in reference_audios:
                if isinstance(audio, str):
                    reference_signals.append(self.audio_evaluator.load_audio(audio))
                else:
                    reference_signals.append(audio)
            
            generated_signals = []
            for audio in generated_audios:
                if isinstance(audio, str):
                    generated_signals.append(self.audio_evaluator.load_audio(audio))
                else:
                    generated_signals.append(audio)
            
            if len(reference_signals) > self.max_audio_samples or len(generated_signals) > self.max_audio_samples:
                print(f"⚠️  Limiting audio evaluation to {self.max_audio_samples} samples due to resource mode '{self.resource_mode}'")
                reference_signals = reference_signals[:self.max_audio_samples]
                generated_signals = generated_signals[:self.max_audio_samples]
            
            results = self.audio_evaluator.evaluate_audio_model(reference_signals, generated_signals)
            results['model_name'] = model_name
            results['model_type'] = 'audio'
            results['timestamp'] = datetime.now().isoformat()
            
            self.evaluation_results[model_name] = results
            
            report = self.audio_evaluator.generate_report(results)
            print(report)
            
            return results
            
        except Exception as e:
            print(f"Error evaluating audio model: {e}")
            return {}
    
    def compare_models(self, model_names: List[str]) -> pd.DataFrame:
        comparison_data = []
        
        for model_name in model_names:
            if model_name in self.evaluation_results:
                results = self.evaluation_results[model_name]
                
                row = {
                    'Model Name': model_name,
                    'Model Type': results.get('model_type', 'unknown'),
                    'Overall Score': results.get('overall_score', 0.0),
                    'Timestamp': results.get('timestamp', '')
                }
                
                if results.get('model_type') == 'text':
                    row['BLEU Score'] = results.get('bleu', {}).get('bleu_score', 0.0)
                    row['ROUGE-L F1'] = results.get('rouge', {}).get('rougeL', {}).get('fmeasure', 0.0)
                    row['BERTScore F1'] = results.get('bert_score', {}).get('bert_score_f1', 0.0)
                
                elif results.get('model_type') == 'image':
                    row['FID Score'] = results.get('fid_score', float('inf'))
                    row['Inception Score'] = results.get('inception_score', {}).get('mean', 0.0)
                
                elif results.get('model_type') == 'audio':
                    row['STOI Score'] = results.get('stoi', {}).get('mean', 0.0)
                    row['SNR Score'] = results.get('snr', {}).get('mean', 0.0)
                
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def save_results(self, filepath: str, format: str = 'json'):
        try:
            if format.lower() == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
            
            elif format.lower() == 'csv':
                flattened_data = []
                for model_name, results in self.evaluation_results.items():
                    flat_row = {'model_name': model_name}
                    self._flatten_dict(results, flat_row)
                    flattened_data.append(flat_row)
                
                df = pd.DataFrame(flattened_data)
                df.to_csv(filepath, index=False)
            
            print(f"Results saved to {filepath}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def _flatten_dict(self, d: Dict, flat_dict: Dict, parent_key: str = '', sep: str = '_'):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                self._flatten_dict(v, flat_dict, new_key, sep=sep)
            else:
                flat_dict[new_key] = v
    
    def load_results(self, filepath: str, format: str = 'json'):
        try:
            if format.lower() == 'json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.evaluation_results = json.load(f)
            
            elif format.lower() == 'csv':
                df = pd.read_csv(filepath)
                self.evaluation_results = {}
                for _, row in df.iterrows():
                    model_name = row['model_name']
                    self.evaluation_results[model_name] = row.to_dict()
            
            print(f"Results loaded from {filepath}")
            
        except Exception as e:
            print(f"Error loading results: {e}")
    
    def generate_summary_report(self) -> str:
        if not self.evaluation_results:
            return "No evaluation results available."
        
        report = "=== GENAI MODEL EVALUATION SUMMARY ===\n\n"
        
        text_models = {k: v for k, v in self.evaluation_results.items() if v.get('model_type') == 'text'}
        image_models = {k: v for k, v in self.evaluation_results.items() if v.get('model_type') == 'image'}
        audio_models = {k: v for k, v in self.evaluation_results.items() if v.get('model_type') == 'audio'}
        
        if text_models:
            report += f"TEXT MODELS ({len(text_models)}):\n"
            for model_name, results in text_models.items():
                report += f"  {model_name}: Overall Score = {results.get('overall_score', 0.0):.4f}\n"
            report += "\n"
        
        if image_models:
            report += f"IMAGE MODELS ({len(image_models)}):\n"
            for model_name, results in image_models.items():
                report += f"  {model_name}: Overall Score = {results.get('overall_score', 0.0):.4f}\n"
            report += "\n"
        
        if audio_models:
            report += f"AUDIO MODELS ({len(audio_models)}):\n"
            for model_name, results in audio_models.items():
                report += f"  {model_name}: Overall Score = {results.get('overall_score', 0.0):.4f}\n"
            report += "\n"
        
        total_models = len(self.evaluation_results)
        avg_score = sum(r.get('overall_score', 0.0) for r in self.evaluation_results.values()) / total_models
        
        report += f"TOTAL MODELS EVALUATED: {total_models}\n"
        report += f"AVERAGE OVERALL SCORE: {avg_score:.4f}\n"
        
        return report


# ----------------------------
# CLI
# ----------------------------
def _cli():
    import argparse, os, json, sys

    parser = argparse.ArgumentParser(description="Unified GenAI evaluator")
    parser.add_argument("--mode", choices=["text","image","audio","all"], default="all")
    parser.add_argument("--real_images", type=str, help="Dir with real images")
    parser.add_argument("--gen_images", type=str, help="Dir with generated images")
    parser.add_argument("--refs_txt", type=str, help="File with reference texts (.txt or .json)")
    parser.add_argument("--cands_txt", type=str, help="File with candidate texts (.txt or .json)")
    parser.add_argument("--refs_audio", type=str, help="Glob for reference wavs")
    parser.add_argument("--cands_audio", type=str, help="Glob for candidate wavs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--per_sample", action="store_true")
    parser.add_argument("--use_comet", action="store_true")
    args = parser.parse_args()


    from .utils.profiler import Profiler
    prof_stats = {}
    evaluator = GenAIModelEvaluator(device=args.device)
    out = {}

    if args.mode in ("image","all") and args.gen_images:
        from .image_evaluator import load_images_from_dir
        real = load_images_from_dir(args.real_images) if args.real_images else []
        gen = load_images_from_dir(args.gen_images) if args.gen_images else []
        if args.cache_dir:
            evaluator.image_evaluator.cache_dir = args.cache_dir
        if real and gen:
            with Profiler("image_fid") as P:
                out["image_fid"] = evaluator.image_evaluator.calculate_fid(real, gen)
            prof_stats["image_fid"] = P.to_dict()
        if gen:
            with Profiler("image_inception") as P2:
                is_mean, is_std = evaluator.image_evaluator.calculate_inception_score(gen)
            prof_stats["image_inception"] = P2.to_dict()
            out["image_inception"] = {"mean": is_mean, "std": is_std}

    if args.mode in ("text","all") and args.refs_txt and args.cands_txt:
        import pathlib
        def _load(path: str):
            p = pathlib.Path(path)
            if p.suffix.lower() == ".json":
                return json.loads(p.read_text(encoding="utf-8"))
            return [line.rstrip("\n") for line in p.read_text(encoding="utf-8").splitlines()]
        refs = _load(args.refs_txt)
        cands = _load(args.cands_txt)
        with Profiler("text_eval") as PT:
            evaluator.text_evaluator.cache_dir = args.cache_dir
        evaluator.text_evaluator.use_comet = args.use_comet
        with Profiler("text_eval") as PT:
            tres = evaluator.text_evaluator.evaluate(refs, cands, per_sample=args.per_sample)
        prof_stats["text_eval"] = PT.to_dict()
        prof_stats["text_eval"] = PT.to_dict()
        out["text"] = {
            "bleu": tres.bleu,
            "sacrebleu": tres.sacrebleu,
            "rouge1_f": tres.rouge1_f,
            "rouge2_f": tres.rouge2_f,
            "rougeL_f": tres.rougeL_f,
            "length_ratio_mean": tres.length_ratio_mean,
            "length_ratio_std": tres.length_ratio_std,
            "overall_score": tres.overall_score,
        }

    if args.mode in ("audio","all") and args.refs_audio and args.cands_audio:
        from .audio_evaluator import AudioEvaluator
        import glob
        ev = AudioEvaluator(cache_dir=args.cache_dir)
        with Profiler("audio_eval") as PA:
            res = ev.evaluate_pairs(sorted(glob.glob(args.refs_audio)),
                                sorted(glob.glob(args.cands_audio)), per_sample=args.per_sample)
        prof_stats["audio_eval"] = PA.to_dict()
        out["audio"] = {
            "snr_mean": res.snr_mean,
            "snr_std": res.snr_std,
            "stoi_mean": res.stoi_mean,
            "stoi_std": res.stoi_std,
        }

    if not out:
        print("Nothing to do. Provide proper inputs.")
        sys.exit(2)

    if args.profile:
        out["_profile"] = prof_stats
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    _cli()
