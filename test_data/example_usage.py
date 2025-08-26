#!/usr/bin/env python3
"""
Example: Using Generated Test Data

This script demonstrates how to use the generated test data
with the GenAI evaluation framework.
"""

from test_data.data_loader import load_text_data, load_image_data, load_audio_data
from src.main_evaluator import GenAIModelEvaluator

def main():
    print("🚀 Example: Using Generated Test Data")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = GenAIModelEvaluator()
    
    # Load test data
    print("\n📝 Loading text data...")
    text_refs, text_cands = load_text_data()
    print(f"Loaded {len(text_refs)} text samples")
    
    print("\n🖼️ Loading image data...")
    img_refs, img_cands = load_image_data()
    print(f"Loaded {len(img_refs)} image samples")
    
    print("\n🎵 Loading audio data...")
    audio_refs, audio_cands = load_audio_data()
    print(f"Loaded {len(audio_refs)} audio samples")
    
    # Run evaluations
    print("\n🔍 Running evaluations...")
    
    # Text evaluation
    text_results = evaluator.evaluate_text_model(
        references=text_refs,
        candidates=text_cands,
        model_name="test_text_model"
    )
    print(f"✓ Text evaluation completed")
    
    # Image evaluation
    image_results = evaluator.evaluate_image_model(
        real_images=img_refs,
        generated_images=img_cands,
        model_name="test_image_model"
    )
    print(f"✓ Image evaluation completed")
    
    # Audio evaluation
    audio_results = evaluator.evaluate_audio_model(
        reference_audios=audio_refs,
        generated_audios=audio_cands,
        model_name="test_audio_model"
    )
    print(f"✓ Audio evaluation completed")
    
    # Save results
    all_results = {
        "text_model": text_results,
        "image_model": image_results,
        "audio_model": audio_results
    }
    
    evaluator.save_results(all_results, "test_data/results/example_evaluation.json")
    print("\n💾 Results saved to: test_data/results/example_evaluation.json")
    
    # Generate summary report
    evaluator.generate_summary_report(all_results, "test_data/results/example_summary.txt")
    print("📊 Summary report saved to: test_data/results/example_summary.txt")
    
    print("\n✅ Example completed successfully!")

if __name__ == "__main__":
    main()
