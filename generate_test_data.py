#!/usr/bin/env python3

import os
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import soundfile as sf
from datetime import datetime
import random
import string

def create_test_directories():
    directories = [
        'test_data',
        'test_data/text',
        'test_data/images',
        'test_data/audio',
        'test_data/results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def generate_text_data():
    print("\n📝 Generating text test data...")
    
    references = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models can process complex patterns in data.",
        "Computer vision allows machines to interpret visual information.",
        "Neural networks are inspired by biological brain structures.",
        "Transfer learning enables models to apply knowledge from one task to another.",
        "Reinforcement learning involves learning through interaction with an environment.",
        "Generative adversarial networks can create realistic synthetic data.",
        "Transformer models have revolutionized natural language processing."
    ]
    
    candidates = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is part of artificial intelligence.",
        "NLP helps computers understand human language.",
        "Deep learning can process complex data patterns.",
        "Computer vision lets machines interpret visual data.",
        "Neural networks mimic biological brain structures.",
        "Transfer learning helps models use knowledge across tasks.",
        "Reinforcement learning learns through environment interaction.",
        "GANs can generate realistic synthetic data.",
        "Transformers have changed natural language processing."
    ]
    
    text_data = {
        'references': references,
        'candidates': candidates,
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'num_samples': len(references),
            'description': 'Sample text data for GenAI evaluation'
        }
    }
    
    with open('test_data/text/sample_text_data.json', 'w', encoding='utf-8') as f:
        json.dump(text_data, f, indent=2, ensure_ascii=False)
    
    df = pd.DataFrame({
        'reference': references,
        'candidate': candidates,
        'sample_id': range(1, len(references) + 1)
    })
    df.to_csv('test_data/text/sample_text_data.csv', index=False)
    
    print(f"✓ Generated {len(references)} text samples")
    print("✓ Saved to: test_data/text/sample_text_data.json")
    print("✓ Saved to: test_data/text/sample_text_data.csv")
    
    return references, candidates

def generate_image_data():
    print("\n🖼️ Generating image test data...")
    
    def create_synthetic_image(width=224, height=224, complexity='simple'):
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        if complexity == 'simple':
            draw.rectangle([50, 50, width-50, height-50], outline='black', width=3)
            draw.ellipse([75, 75, width-75, height-75], fill='lightblue')
            draw.text((width//2-20, height//2-10), "AI", fill='black')
            
        elif complexity == 'medium':
            for i in range(0, width, 30):
                draw.line([(i, 0), (i, height)], fill='gray', width=1)
            for j in range(0, height, 30):
                draw.line([(0, j), (width, j)], fill='gray', width=1)
            draw.rectangle([60, 60, width-60, height-60], fill='lightgreen')
            draw.text((width//2-30, height//2-10), "GENAI", fill='darkgreen')
            
        else:
            colors = ['red', 'blue', 'green', 'yellow', 'purple']
            for i in range(5):
                x1 = random.randint(20, width-40)
                y1 = random.randint(20, height-40)
                x2 = x1 + random.randint(30, 80)
                y2 = y1 + random.randint(30, 80)
                draw.rectangle([x1, y1, x2, y2], fill=random.choice(colors))
            
            for _ in range(3):
                x = random.randint(30, width-30)
                y = random.randint(30, height-30)
                radius = random.randint(10, 30)
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                           fill=random.choice(colors))
        
        return img
    
    reference_images = []
    for i in range(10):
        complexity = random.choice(['simple', 'medium', 'complex'])
        img = create_synthetic_image(complexity=complexity)
        reference_images.append(img)
        
        img.save(f'test_data/images/reference_{i+1:02d}.png')
    
    candidate_images = []
    for i in range(10):
        if i < 3:
            img = reference_images[i].copy()
        else:
            img = create_synthetic_image(complexity=random.choice(['simple', 'medium', 'complex']))
        
        candidate_images.append(img)
        img.save(f'test_data/images/candidate_{i+1:02d}.png')
    
    image_metadata = {
        'num_samples': len(reference_images),
        'image_size': '224x224',
        'format': 'PNG',
        'generated_at': datetime.now().isoformat(),
        'description': 'Synthetic images for GenAI evaluation'
    }
    
    with open('test_data/images/image_metadata.json', 'w') as f:
        json.dump(image_metadata, f, indent=2)
    
    print(f"✓ Generated {len(reference_images)} image pairs")
    print("✓ Saved to: test_data/images/")
    print("✓ Saved metadata to: test_data/images/image_metadata.json")
    
    return reference_images, candidate_images

def generate_audio_data():
    print("\n🎵 Generating audio test data...")
    
    sample_rate = 22050
    duration = 3.0
    num_samples = int(sample_rate * duration)
    
    def create_synthetic_audio(complexity='simple'):
        t = np.linspace(0, duration, num_samples)
        
        if complexity == 'simple':
            frequency = 440
            audio = np.sin(2 * np.pi * frequency * t)
            
        elif complexity == 'medium':
            audio = (np.sin(2 * np.pi * 440 * t) + 
                    0.5 * np.sin(2 * np.pi * 880 * t) +
                    0.3 * np.sin(2 * np.pi * 220 * t))
            
        else:
            audio = (np.sin(2 * np.pi * 440 * t) +
                    0.7 * np.sin(2 * np.pi * 660 * t) +
                    0.4 * np.sin(2 * np.pi * 880 * t))
            noise = np.random.normal(0, 0.1, num_samples)
            audio += noise
        
        audio = audio / np.max(np.abs(audio))
        return audio
    
    reference_audios = []
    for i in range(10):
        complexity = random.choice(['simple', 'medium', 'complex'])
        audio = create_synthetic_audio(complexity=complexity)
        reference_audios.append(audio)
        
        sf.write(f'test_data/audio/reference_{i+1:02d}.wav', audio, sample_rate)
    
    candidate_audios = []
    for i in range(10):
        if i < 3:
            audio = reference_audios[i].copy()
        else:
            complexity = random.choice(['simple', 'medium', 'complex'])
            audio = create_synthetic_audio(complexity=complexity)
        
        candidate_audios.append(audio)
        sf.write(f'test_data/audio/candidate_{i+1:02d}.wav', audio, sample_rate)
    
    audio_metadata = {
        'num_samples': len(reference_audios),
        'sample_rate': sample_rate,
        'duration': duration,
        'format': 'WAV',
        'generated_at': datetime.now().isoformat(),
        'description': 'Synthetic audio for GenAI evaluation'
    }
    
    with open('test_data/audio/audio_metadata.json', 'w') as f:
        json.dump(audio_metadata, f, indent=2)
    
    print(f"✓ Generated {len(reference_audios)} audio pairs")
    print("✓ Saved to: test_data/audio/")
    print("✓ Saved metadata to: test_data/audio/audio_metadata.json")
    
    return reference_audios, candidate_audios

def create_data_loader():
    loader_code = '''#!/usr/bin/env python3
"""
Test Data Loader for GenAI Metrics Evaluation

This module provides easy access to the generated test data.
"""

import json
import numpy as np
from PIL import Image
import soundfile as sf
import os

class TestDataLoader:
    """Loader for test data files."""
    
    def __init__(self, data_dir='test_data'):
        self.data_dir = data_dir
    
    def load_text_data(self):
        """Load text test data."""
        with open(f'{self.data_dir}/text/sample_text_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['references'], data['candidates']
    
    def load_image_data(self):
        """Load image test data."""
        references = []
        candidates = []
        
        # Load reference images
        for i in range(1, 11):
            img_path = f'{self.data_dir}/images/reference_{i:02d}.png'
            if os.path.exists(img_path):
                references.append(Image.open(img_path))
        
        # Load candidate images
        for i in range(1, 11):
            img_path = f'{self.data_dir}/images/candidate_{i:02d}.png'
            if os.path.exists(img_path):
                candidates.append(Image.open(img_path))
        
        return references, candidates
    
    def load_audio_data(self):
        """Load audio test data."""
        references = []
        candidates = []
        
        # Load reference audio
        for i in range(1, 11):
            audio_path = f'{self.data_dir}/audio/reference_{i:02d}.wav'
            if os.path.exists(audio_path):
                audio, sr = sf.read(audio_path)
                references.append(audio)
        
        # Load candidate audio
        for i in range(1, 11):
            audio_path = f'{self.data_dir}/audio/candidate_{i:02d}.wav'
            if os.path.exists(audio_path):
                audio, sr = sf.read(audio_path)
                candidates.append(audio)
        
        return references, candidates
    
    def get_metadata(self, data_type):
        """Get metadata for specific data type."""
        metadata_path = f'{self.data_dir}/{data_type}/{data_type}_metadata.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None

# Convenience functions
def load_text_data():
    """Load text test data."""
    loader = TestDataLoader()
    return loader.load_text_data()

def load_image_data():
    """Load image test data."""
    loader = TestDataLoader()
    return loader.load_image_data()

def load_audio_data():
    """Load audio test data."""
    loader = TestDataLoader()
    return loader.load_audio_data()
'''
    
    with open('test_data/data_loader.py', 'w', encoding='utf-8') as f:
        f.write(loader_code)
    
    print("✓ Created data loader: test_data/data_loader.py")

def create_usage_example():
    example_code = '''#!/usr/bin/env python3
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
    print("\\n📝 Loading text data...")
    text_refs, text_cands = load_text_data()
    print(f"Loaded {len(text_refs)} text samples")
    
    print("\\n🖼️ Loading image data...")
    img_refs, img_cands = load_image_data()
    print(f"Loaded {len(img_refs)} image samples")
    
    print("\\n🎵 Loading audio data...")
    audio_refs, audio_cands = load_audio_data()
    print(f"Loaded {len(audio_refs)} audio samples")
    
    # Run evaluations
    print("\\n🔍 Running evaluations...")
    
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
    print("\\n💾 Results saved to: test_data/results/example_evaluation.json")
    
    # Generate summary report
    evaluator.generate_summary_report(all_results, "test_data/results/example_summary.txt")
    print("📊 Summary report saved to: test_data/results/example_summary.txt")
    
    print("\\n✅ Example completed successfully!")

if __name__ == "__main__":
    main()
'''
    
    with open('test_data/example_usage.py', 'w', encoding='utf-8') as f:
        f.write(example_code)
    
    print("✓ Created usage example: test_data/example_usage.py")

def main():
    print("🎯 GenAI Test Data Generator")
    print("=" * 40)
    
    create_test_directories()
    
    text_refs, text_cands = generate_text_data()
    img_refs, img_cands = generate_image_data()
    audio_refs, audio_cands = generate_audio_data()
    
    create_data_loader()
    create_usage_example()
    
    readme_content = '''# Test Data Directory

This directory contains generated test data for the GenAI Metrics Evaluation project.

## Structure

```
test_data/
├── text/                    # Text evaluation data
│   ├── sample_text_data.json
│   └── sample_text_data.csv
├── images/                  # Image evaluation data
│   ├── reference_01.png - reference_10.png
│   ├── candidate_01.png - candidate_10.png
│   └── image_metadata.json
├── audio/                   # Audio evaluation data
│   ├── reference_01.wav - reference_10.wav
│   ├── candidate_01.wav - candidate_10.wav
│   └── audio_metadata.json
├── results/                 # Evaluation results (generated when running examples)
├── data_loader.py          # Utility to load test data
└── example_usage.py        # Example script showing how to use the data
```

## Usage

### Quick Start
```python
from test_data.data_loader import load_text_data, load_image_data, load_audio_data

# Load test data
text_refs, text_cands = load_text_data()
img_refs, img_cands = load_image_data()
audio_refs, audio_cands = load_audio_data()
```

### Run Example
```bash
python test_data/example_usage.py
```

## Data Description

- **Text Data**: 10 sample text pairs with varying similarity levels
- **Image Data**: 10 synthetic image pairs (224x224 PNG format)
- **Audio Data**: 10 synthetic audio pairs (3-second WAV files, 22.05kHz)

## Regenerating Data

To regenerate all test data, run:
```bash
python generate_test_data.py
```

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
'''
    readme_content = '''# Test Data Directory

This directory contains generated test data for the GenAI Metrics Evaluation project.

## Structure

```
test_data/
├── text/                    # Text evaluation data
│   ├── sample_text_data.json
│   └── sample_text_data.csv
├── images/                  # Image evaluation data
│   ├── reference_01.png - reference_10.png
│   ├── candidate_01.png - candidate_10.png
│   └── image_metadata.json
├── audio/                   # Audio evaluation data
│   ├── reference_01.wav - reference_10.wav
│   ├── candidate_01.wav - candidate_10.wav
│   └── audio_metadata.json
├── results/                 # Evaluation results (generated when running examples)
├── data_loader.py          # Utility to load test data
└── example_usage.py        # Example script showing how to use the data
```

## Usage

### Quick Start
```python
from test_data.data_loader import load_text_data, load_image_data, load_audio_data

# Load test data
text_refs, text_cands = load_text_data()
img_refs, img_cands = load_image_data()
audio_refs, audio_cands = load_audio_data()
```

### Run Example
```bash
python test_data/example_usage.py
```

## Data Description

- **Text Data**: 10 sample text pairs with varying similarity levels
- **Image Data**: 10 synthetic image pairs (224x224 PNG format)
- **Audio Data**: 10 synthetic audio pairs (3-second WAV files, 22.05kHz)

## Regenerating Data

To regenerate all test data, run:
```bash
python generate_test_data.py
```

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
'''
    
    with open('test_data/README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("\n📋 Created documentation: test_data/README.md")
    
    print("\n🎉 Test data generation completed!")
    print("\n📁 Generated files:")
    print("  • test_data/text/ - Text evaluation data")
    print("  • test_data/images/ - Image evaluation data") 
    print("  • test_data/audio/ - Audio evaluation data")
    print("  • test_data/data_loader.py - Data loading utility")
    print("  • test_data/example_usage.py - Usage example")
    print("  • test_data/README.md - Documentation")
    
    print("\n🚀 Next steps:")
    print("  1. Run: python test_data/example_usage.py")
    print("  2. Or use the data loader in your own scripts")
    print("  3. Check test_data/README.md for more details")

if __name__ == "__main__":
    main()
