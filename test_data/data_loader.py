#!/usr/bin/env python3
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
