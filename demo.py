import numpy as np
from PIL import Image
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main_evaluator import GenAIModelEvaluator

try:
    from test_data.data_loader import load_text_data, load_image_data, load_audio_data
    TEST_DATA_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: test_data not found. Please run 'python generate_test_data.py' first.")
    TEST_DATA_AVAILABLE = False

def create_sample_text_data():
    references = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models require large amounts of training data.",
        "Computer vision algorithms can identify objects in images."
    ]
    
    candidates = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is part of artificial intelligence.",
        "NLP helps computers understand human language.",
        "Deep learning needs lots of data to train.",
        "Vision systems can detect objects in pictures."
    ]
    
    return references, candidates

def create_sample_image_data():
    real_images = []
    generated_images = []
    
    for i in range(5):
        real_img = Image.new('RGB', (64, 64), color=(i * 50, 100, 150))
        real_images.append(real_img)
        
        gen_img = Image.new('RGB', (64, 64), color=(i * 50 + 10, 110, 160))
        generated_images.append(gen_img)
    
    return real_images, generated_images

def create_sample_audio_data():
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    reference_audios = []
    generated_audios = []
    
    for i in range(5):
        freq = 440 + i * 100
        ref_signal = np.sin(2 * np.pi * freq * t)
        reference_audios.append(ref_signal)
        
        gen_signal = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
        generated_audios.append(gen_signal)
    
    return reference_audios, generated_audios

def load_test_data_or_fallback():
    if TEST_DATA_AVAILABLE:
        try:
            print("📁 Loading generated test data...")
            text_refs, text_cands = load_text_data()
            img_refs, img_cands = load_image_data()
            audio_refs, audio_cands = load_audio_data()
            
            print(f"✓ Loaded {len(text_refs)} text samples")
            print(f"✓ Loaded {len(img_refs)} image samples")
            print(f"✓ Loaded {len(audio_refs)} audio samples")
            
            return text_refs, text_cands, img_refs, img_cands, audio_refs, audio_cands
            
        except Exception as e:
            print(f"⚠️  Error loading test data: {e}")
            print("🔄 Falling back to sample data...")
            return load_fallback_data()
    else:
        print("🔄 Using fallback sample data...")
        return load_fallback_data()

def load_fallback_data():
    text_refs, text_cands = create_sample_text_data()
    img_refs, img_cands = create_sample_image_data()
    audio_refs, audio_cands = create_sample_audio_data()
    return text_refs, text_cands, img_refs, img_cands, audio_refs, audio_cands

def demo_text_evaluation():
    print("=" * 50)
    print("TEXT MODEL EVALUATION DEMO")
    print("=" * 50)
    
    evaluator = GenAIModelEvaluator()
    
    text_refs, text_cands, _, _, _, _ = load_test_data_or_fallback()
    
    print(f"Evaluating {len(text_refs)} text samples...")
    
    results = evaluator.evaluate_text_model(
        references=text_refs,
        candidates=text_cands,
        model_name="demo_text_model"
    )
    
    return results

def demo_image_evaluation():
    print("=" * 50)
    print("IMAGE MODEL EVALUATION DEMO")
    print("=" * 50)
    
    evaluator = GenAIModelEvaluator()
    
    _, _, img_refs, img_cands, _, _ = load_test_data_or_fallback()
    
    print(f"Evaluating {len(img_refs)} image samples...")
    
    results = evaluator.evaluate_image_model(
        real_images=img_refs,
        generated_images=img_cands,
        model_name="demo_image_model"
    )
    
    return results

def demo_audio_evaluation():
    print("=" * 50)
    print("AUDIO MODEL EVALUATION DEMO")
    print("=" * 50)
    
    evaluator = GenAIModelEvaluator()
    
    _, _, _, _, audio_refs, audio_cands = load_test_data_or_fallback()
    
    print(f"Evaluating {len(audio_refs)} audio samples...")
    
    results = evaluator.evaluate_audio_model(
        reference_audios=audio_refs,
        generated_audios=audio_cands,
        model_name="demo_audio_model"
    )
    
    return results

def demo_comprehensive_evaluation():
    print("=" * 50)
    print("COMPREHENSIVE EVALUATION DEMO")
    print("=" * 50)
    
    evaluator = GenAIModelEvaluator()
    
    text_refs, text_cands, img_refs, img_cands, audio_refs, audio_cands = load_test_data_or_fallback()
    
    print("\n📝 Evaluating text model...")
    text_results = evaluator.evaluate_text_model(
        references=text_refs,
        candidates=text_cands,
        model_name="demo_text_model"
    )
    
    print("\n🖼️ Evaluating image model...")
    image_results = evaluator.evaluate_image_model(
        real_images=img_refs,
        generated_images=img_cands,
        model_name="demo_image_model"
    )
    
    print("\n🎵 Evaluating audio model...")
    audio_results = evaluator.evaluate_audio_model(
        reference_audios=audio_refs,
        generated_audios=audio_cands,
        model_name="demo_audio_model"
    )
    
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    
    comparison_df = evaluator.compare_models([
        "demo_text_model",
        "demo_image_model", 
        "demo_audio_model"
    ])
    
    print(comparison_df.to_string(index=False))
    
    print("\n" + "=" * 50)
    print("SUMMARY REPORT")
    print("=" * 50)
    
    summary = evaluator.generate_summary_report()
    print(summary)
    
    print("\n" + "=" * 50)
    print("SAVING RESULTS")
    print("=" * 50)
    
    evaluator.save_results("evaluation_results.json", format="json")
    evaluator.save_results("evaluation_results.csv", format="csv")
    
    # Generate HTML report
    print("\n📊 Generating HTML report...")
    try:
        from tools.report_builder import build_report
        build_report("evaluation_results.json", "report.html", top_k=5)
        print("✓ HTML report generated: report.html")
    except Exception as e:
        print(f"⚠️  Warning: Could not generate HTML report: {e}")
    
    return evaluator

def demo_custom_evaluation():
    print("=" * 50)
    print("CUSTOM EVALUATION SCENARIOS")
    print("=" * 50)
    
    evaluator = GenAIModelEvaluator(resource_mode='lightweight')
    
    print("\nScenario 1: Multiple Text Models")
    
    if TEST_DATA_AVAILABLE:
        try:
            text_refs, text_cands, _, _, _, _ = load_test_data_or_fallback()
            references = text_refs[:3]
            high_quality_candidates = text_cands[:3]
            medium_quality_candidates = [ref.replace("the", "a").replace("is", "was") for ref in references]
            low_quality_candidates = [ref.split()[0] + " " + ref.split()[-1] for ref in references]
        except:
            references = [
                "The weather is beautiful today.",
                "I love programming in Python.",
                "Artificial intelligence is transforming industries."
            ]
            
            high_quality_candidates = [
                "The weather is beautiful today.",
                "I love programming in Python.",
                "Artificial intelligence is transforming industries."
            ]
            
            medium_quality_candidates = [
                "The weather is nice today.",
                "I enjoy programming in Python.",
                "AI is changing industries."
            ]
            
            low_quality_candidates = [
                "Weather good.",
                "Python programming.",
                "AI changes things."
            ]
    else:
        references = [
            "The weather is beautiful today.",
            "I love programming in Python.",
            "Artificial intelligence is transforming industries."
        ]
        
        high_quality_candidates = [
            "The weather is beautiful today.",
            "I love programming in Python.",
            "Artificial intelligence is transforming industries."
        ]
        
        medium_quality_candidates = [
            "The weather is nice today.",
            "I enjoy programming in Python.",
            "AI is changing industries."
        ]
        
        low_quality_candidates = [
            "Weather good.",
            "Python programming.",
            "AI changes things."
        ]
    
    evaluator.evaluate_text_model(references, high_quality_candidates, "high_quality_text")
    evaluator.evaluate_text_model(references, medium_quality_candidates, "medium_quality_text")
    evaluator.evaluate_text_model(references, low_quality_candidates, "low_quality_text")
    
    comparison_df = evaluator.compare_models([
        "high_quality_text",
        "medium_quality_text",
        "low_quality_text"
    ])
    
    print(comparison_df.to_string(index=False))
    
    return evaluator

def main():
    print("GenAI Model Evaluation Framework Demo")
    print("=" * 60)
    print("💡 Using lightweight mode to prevent excessive resource usage.")
    print("   For full evaluation, modify resource_mode='full' in the code.")
    
    if not TEST_DATA_AVAILABLE:
        print("\n💡 Tip: Run 'python generate_test_data.py' to create test data for better demonstrations.")
    
    try:
        evaluator = demo_comprehensive_evaluation()
        
        demo_custom_evaluation()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nFiles created:")
        print("- evaluation_results.json")
        print("- evaluation_results.csv")
        print("- report.html")
        print("\nYou can now use these results for further analysis.")
        print("📊 Open report.html in your browser to view the detailed evaluation report.")
        
        if TEST_DATA_AVAILABLE:
            print("\n✅ Demo used generated test data from test_data/ directory.")
        else:
            print("\n⚠️  Demo used fallback sample data. Consider generating test data for better results.")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Please check that all required dependencies are installed.")
        if not TEST_DATA_AVAILABLE:
            print("Also ensure test data is generated by running: python generate_test_data.py")

if __name__ == "__main__":
    main()
