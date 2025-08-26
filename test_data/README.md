# Test Data Directory

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
