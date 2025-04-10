#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepFake Video Analysis Tool
A simplified interface to the deepfake detection engine
"""

import os
import sys
import argparse
from deepfake_detection_engine import init, process

def analyze_video(video_path, models_path="weights", config_path="config.json", device="cpu"):
    """
    Analyze a video file for deepfake content
    
    Args:
        video_path: Path to video file
        models_path: Path to model weights
        config_path: Path to configuration file
        device: Device to run inference on (cpu/cuda)
        
    Returns:
        Deepfake score and classification
    """
    # Initialize models
    init(models_path, config_path, device)
    
    # Process video
    score = process(video_path)
    
    return {
        "score": score,
        "is_fake": score > 0.5,
        "confidence": abs(score - 0.5) * 2
    }

def print_results(results):
    """Print analysis results in a formatted way"""
    print("\n" + "="*50)
    print(" DEEPFAKE ANALYSIS RESULTS")
    print("="*50)
    
    # Classification result
    print(f"\nCLASSIFICATION: {'⚠️ FAKE' if results['is_fake'] else '✓ AUTHENTIC'}")
    
    # Score details
    print(f"\nDetection Score: {results['score']:.4f} (0=real, 1=fake)")
    print(f"Confidence: {results['confidence']:.2%}")
    
    # Interpretation
    if results['is_fake']:
        if results['score'] > 0.8:
            print("\nINTERPRETATION: High confidence this video contains deepfake content")
        else:
            print("\nINTERPRETATION: This video likely contains manipulated content")
    else:
        if results['score'] < 0.2:
            print("\nINTERPRETATION: High confidence this video is authentic")
        else:
            print("\nINTERPRETATION: This video appears to be authentic")
    
    print("\n" + "="*50 + "\n")

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="DeepFake Video Analysis Tool")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--models", default="weights", help="Path to model weights")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for inference")
    
    args = parser.parse_args()
    
    # Validate video path
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Run analysis
    try:
        results = analyze_video(args.video, args.models, args.config, args.device)
        print_results(results)
    except Exception as e:
        print(f"Error analyzing video: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 