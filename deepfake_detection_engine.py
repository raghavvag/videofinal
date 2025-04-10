#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepFake Detection System
Core scanning and detection module
"""

# Core libraries
import os
import sys
import glob
import json
import time
import warnings
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings("ignore")

# Data processing and ML libraries
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ML components
from dface import MTCNN, FaceNet
from sklearn.cluster import DBSCAN
import timm.models.efficientnet as effnet

# --------------------------------------
# Global Configuration
# --------------------------------------
CONFIG = {
    "device": "cpu",
    "margin": 0,
    "fps": 1,
    "batch": 32,
    "face_dimensions": None
}

# --------------------------------------
# Models
# --------------------------------------
DETECTION_MODELS = {
    "face_detector": None,  # MTCNN
    "feature_extractor": None,  # FaceNet
    "classifier": None   # DeepWare
}

# Image preprocessing transformer
IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# --------------------------------------
# Model Architecture
# --------------------------------------
class DeepFakeDetector(nn.Module):
    def __init__(self, architecture='b3'):
        """EfficientNet-based deepfake detector"""
        super(DeepFakeDetector, self).__init__()
        
        # Architecture size mapping
        size_mapping = {
            'b1': 1280, 'b2': 1408, 'b3': 1536, 
            'b4': 1792, 'b5': 2048, 'b6': 2304, 'b7': 2560
        }
        
        # Validate architecture
        if architecture not in size_mapping:
            raise ValueError(f"Architecture {architecture} not supported")
            
        # Build model components
        backbone = getattr(effnet, f'tf_efficientnet_{architecture}_ns')
        self.encoder = backbone()  # Use encoder to match original weights
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Use avg_pool to match original weights
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(size_mapping[architecture], 1)  # Use fc to match original weights

    def forward(self, x):
        """Forward pass through the model"""
        features = self.encoder.forward_features(x)
        pooled = self.avg_pool(features).flatten(1)
        dropped = self.dropout(pooled)
        logits = self.fc(dropped)
        return logits


class ModelEnsemble(nn.Module):
    def __init__(self, model_list):
        """Ensemble of models for prediction"""
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleList(model_list)  # Use models instead of model_collection

    def forward(self, x):
        """Average predictions from all models"""
        preds = []  # Use preds instead of all_predictions
        for i, model in enumerate(self.models):  # Use self.models
            y = model(x)  # Use y instead of prediction
            preds.append(y)
        
        # Average predictions (keep the old variable names for consistency)
        final = torch.mean(torch.stack(preds), dim=0)
        return final


# --------------------------------------
# Video Processing
# --------------------------------------
def extract_video_frames(video_path, frame_batch_size=10, target_frame_rate=1):
    """
    Extract frames from video at specified frame rate
    
    Args:
        video_path: Path to video file
        frame_batch_size: Number of frames to process in a batch
        target_frame_rate: Desired frames per second to extract
        
    Returns:
        Generator yielding batches of frames
    """
    # Open video
    capture = cv2.VideoCapture(video_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Empty video check
    if frame_count <= 0:
        return None
        
    # Calculate frame extraction indices
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    extraction_fps = min(target_frame_rate, video_fps)
    num_frames_to_extract = int(frame_count / video_fps * extraction_fps)
    frame_indices = np.linspace(0, frame_count, num_frames_to_extract, endpoint=False, dtype=int)
    
    # Process frames
    current_batch = []
    for frame_idx in range(frame_count):
        # Skip frames not in our sample indices
        frame_grabbed = capture.grab()
        if frame_idx not in frame_indices:
            continue
            
        # Read the frame
        success, frame = capture.retrieve()
        if not success:
            continue
            
        # Resize large frames
        height, width = frame.shape[:2]
        if width * height > 1920 * 1080:
            scale_factor = 1920 / max(width, height)
            new_dims = (int(width * scale_factor), int(height * scale_factor))
            frame = cv2.resize(frame, new_dims)
            
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_batch.append(frame_rgb)
        
        # Yield batch when full
        if len(current_batch) == frame_batch_size:
            yield current_batch
            current_batch = []
            
    # Yield remaining frames
    if current_batch:
        yield current_batch
        
    # Release resources
    capture.release()


def extract_face(image, bbox, margin_factor=1):
    """
    Extract face from image with margin
    
    Args:
        image: Source image
        bbox: Bounding box [x1, y1, x2, y2]
        margin_factor: Margin factor to add around face
        
    Returns:
        Face image array
    """
    x1, y1, x2, y2 = bbox
    
    # Calculate size with margin
    face_size = int(max(x2 - x1, y2 - y1) * margin_factor)
    
    # Calculate center and new coordinates
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    new_x1 = center_x - face_size // 2
    new_x2 = center_x + face_size // 2
    new_y1 = center_y - face_size // 2
    new_y2 = center_y + face_size // 2
    
    # Crop and return
    face_img = Image.fromarray(image).crop([new_x1, new_y1, new_x2, new_y2])
    return np.asarray(face_img)


def standardize_face_margins(face_images, target_margin):
    """
    Adjust face margins for consistency
    
    Args:
        face_images: List of face images
        target_margin: Target margin to adjust to
        
    Returns:
        List of adjusted face images
    """
    adjusted_faces = []
    
    for face in face_images:
        # Convert to PIL
        face_pil = Image.fromarray(face)
        
        # Calculate new size
        width, height = face_pil.size
        new_size = int(width / CONFIG["margin"] * target_margin)
        
        # Center crop to new size
        adjusted_face = TF.center_crop(face_pil, (new_size, new_size))
        
        # Convert back to numpy
        adjusted_faces.append(np.asarray(adjusted_face))
        
    return adjusted_faces


# --------------------------------------
# Analysis Functions
# --------------------------------------
def group_faces_by_identity(faces):
    """
    Cluster faces by identity using facial embeddings
    
    Args:
        faces: List of face images
        
    Returns:
        Dictionary of clusters
    """
    # Adjust margins if needed for face recognition
    if CONFIG["margin"] != 1.2:
        faces = standardize_face_margins(faces, 1.2)
        
    # Extract facial embeddings
    embeddings = DETECTION_MODELS["feature_extractor"].embedding(faces)
    
    # Cluster embeddings
    minimum_samples = CONFIG["fps"] * 5
    clustering = DBSCAN(eps=0.35, metric='cosine', min_samples=minimum_samples)
    cluster_labels = clustering.fit_predict(embeddings)
    
    # Group faces by cluster
    clusters = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        clusters[label].append(idx)
        
    # Handle outliers (label -1)
    outliers = {0: clusters.pop(-1, [])}
    
    # Return outliers if they form a significant group
    if len(clusters) == 0 and len(outliers[0]) >= minimum_samples:
        return outliers
        
    return clusters


def identity_prediction_strategy(predictions, threshold=0.8):
    """
    Strategy for per-identity predictions
    
    Args:
        predictions: List of predictions for a single identity
        threshold: Threshold for classifying fake/real
        
    Returns:
        Final prediction score
    """
    predictions = np.array(predictions)
    
    # Group predictions
    fake_preds = predictions[predictions >= threshold]
    real_preds = predictions[predictions <= (1 - threshold)]
    
    # Decision logic
    if len(fake_preds) >= int(len(predictions) * 0.9):
        return np.mean(fake_preds)
    if len(real_preds) >= int(len(predictions) * 0.9):
        return np.mean(real_preds)
        
    # Default case
    return np.mean(predictions)


# Helper functions for prediction strategy
def is_confident(preds):
    """Check if predictions are confident"""
    return np.mean(np.abs(preds - 0.5) * 2) >= 0.7


def adjust_high_scores(score):
    """Adjust high scores for better discrimination"""
    return score - np.log10(score) if score >= 0.8 else score


def overall_prediction_strategy(identity_predictions):
    """
    Overall prediction strategy across all identities
    
    Args:
        identity_predictions: Predictions per identity
        
    Returns:
        Final prediction score
    """
    identity_predictions = np.array(identity_predictions)
    max_score = np.max(identity_predictions)
    
    # High deepfake confidence case
    if max_score >= 0.8:
        if is_confident(identity_predictions):
            return adjust_high_scores(max_score)
        return max_score
        
    # High real confidence case
    if is_confident(identity_predictions):
        return np.min(identity_predictions)
        
    # Default case
    return np.mean(identity_predictions)


# --------------------------------------
# Main Processing Functions
# --------------------------------------
def detect_deepfakes(video_path):
    """
    Scan video for faces and predict deepfake probability
    
    Args:
        video_path: Path to video file
        
    Returns:
        List of predictions and face images
    """
    # Extract frames
    frame_batches = extract_video_frames(
        video_path, 
        CONFIG["batch"], 
        CONFIG["fps"]
    )
    
    # Initialize results
    extracted_faces = []
    predictions = []
    
    # Process each batch of frames
    for batch in frame_batches:
        # Detect faces
        detection_results = DETECTION_MODELS["face_detector"].detect(batch)
        
        # Process each frame's results
        for frame_idx, result in enumerate(detection_results):
            if result is None:
                continue
                
            # Extract faces with high confidence
            bboxes, confidence_scores, landmarks = result
            for box_idx, box in enumerate(bboxes):
                if confidence_scores[box_idx] > 0.98:
                    # Extract and process face
                    face = extract_face(batch[frame_idx], box, CONFIG["margin"])
                    face = cv2.normalize(face, None, 0, 255, cv2.NORM_MINMAX)
                    face = cv2.resize(face, CONFIG["face_dimensions"])
                    extracted_faces.append(face)
    
    # No faces found
    if not extracted_faces:
        return None, []
    
    # Predict on extracted faces
    with torch.no_grad():
        batch_size = CONFIG["batch"]
        num_batches = int(np.ceil(len(extracted_faces) / batch_size))
        
        for batch_idx in range(num_batches):
            # Prepare batch
            face_batch = []
            for face in extracted_faces[batch_idx * batch_size:(batch_idx + 1) * batch_size]:
                processed_face = IMAGE_TRANSFORM(face)
                face_batch.append(processed_face)
                
            # Stack and predict
            input_tensor = torch.stack(face_batch)
            with autocast():
                output = DETECTION_MODELS["classifier"](input_tensor.to(CONFIG["device"]))
                
            predictions.append(output)
    
    # Process predictions
    all_predictions = torch.sigmoid(torch.cat(predictions, dim=0))[:, 0].cpu().numpy()
    return list(all_predictions), extracted_faces


def analyze_video(video_path):
    """
    Process a video file and determine deepfake score
    
    Args:
        video_path: Path to video file
        
    Returns:
        Deepfake score (0-1)
    """
    try:
        # Detect faces and get predictions
        predictions, faces = detect_deepfakes(video_path)
        
        # Handle no predictions
        if predictions is None:
            return 0.5
            
        # Cluster faces by identity
        identity_clusters = group_faces_by_identity(faces)
        if len(identity_clusters) == 0:
            return 0.5
            
        # Group predictions by identity
        identity_predictions = defaultdict(list)
        for identity, face_indices in identity_clusters.items():
            for idx in face_indices:
                identity_predictions[identity].append(predictions[idx])
                
        # Get per-identity predictions
        final_identity_scores = [
            identity_prediction_strategy(preds) 
            for preds in identity_predictions.values()
        ]
        
        if len(final_identity_scores) == 0:
            return 0.5
            
        # Get final score
        final_score = overall_prediction_strategy(final_identity_scores)
        
        # Ensure Python float and clip to valid range
        return float(np.clip(final_score, 0.01, 0.99))
        
    except Exception:
        # Return neutral score on error
        return 0.5


def setup_models(models_directory, config_file, device_name):
    """
    Initialize all models and configurations
    
    Args:
        models_directory: Directory containing model weights
        config_file: Path to configuration file
        device_name: Device to run models on (cpu/cuda)
    """
    global CONFIG, DETECTION_MODELS
    
    # Update device
    CONFIG["device"] = device_name
    
    # Resolve paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(models_directory):
        models_directory = os.path.join(base_dir, models_directory)
    if not os.path.isabs(config_file):
        config_file = os.path.join(base_dir, config_file)
    
    # Load configuration
    with open(config_file) as f:
        config_data = json.loads(f.read())
    
    # Update global configuration
    CONFIG["margin"] = config_data['margin']
    CONFIG["face_dimensions"] = (config_data['size'], config_data['size'])
    architecture = config_data['arch']
    
    # Initialize face detection models
    DETECTION_MODELS["face_detector"] = MTCNN(CONFIG["device"])
    DETECTION_MODELS["feature_extractor"] = FaceNet(CONFIG["device"])
    
    # Load classifier models
    if os.path.isdir(models_directory):
        model_paths = glob.glob(os.path.join(models_directory, '*.pt'))
    else:
        model_paths = [models_directory]
        
    # Ensure models exist
    if not model_paths:
        raise FileNotFoundError(f"No model files found in: {models_directory}")
        
    # Load models
    classifier_models = []
    for model_path in model_paths:
        # Create model instance
        model = DeepFakeDetector(architecture)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint)
        del checkpoint
        
        # Add to list
        classifier_models.append(model)
        
    # Create ensemble and set to evaluation mode
    DETECTION_MODELS["classifier"] = ModelEnsemble(classifier_models).eval().to(CONFIG["device"])


def main():
    """Main execution function"""
    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check arguments
    if len(sys.argv) != 5:
        print('Usage: scan.py <scan_dir> <models_dir> <cfg_file> <device>')
        sys.exit(1)
    
    # Parse arguments
    scan_directory = sys.argv[1]
    if not os.path.isabs(scan_directory):
        scan_directory = os.path.join(base_dir, scan_directory)
    models_directory = sys.argv[2]
    config_file = sys.argv[3]
    device_name = sys.argv[4]
    
    # Initialize models
    setup_models(models_directory, config_file, device_name)
    
    # Get files to process
    if os.path.isdir(scan_directory):
        files_to_process = glob.glob(os.path.join(scan_directory, '*'))
    else:
        with open(scan_directory, 'r') as f:
            files_to_process = [line.strip() for line in f.readlines()]
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(
            executor.map(analyze_video, files_to_process), 
            total=len(files_to_process)
        ))
    
    # Save results
    os.makedirs("models/deepware_video", exist_ok=True)
    with open("models/deepware_video/result.txt", "w") as output_file:
        output_file.write(str(results[0]))


# --------------------------------------
# API Compatibility Functions
# --------------------------------------
# These aliases maintain backward compatibility with existing code
def init(models_dir, cfg_file, dev):
    """
    Legacy function for backward compatibility
    Initializes the detection models
    """
    return setup_models(models_dir, cfg_file, dev)

def process(file):
    """
    Legacy function for backward compatibility
    Processes a video file to detect deepfakes
    """
    return analyze_video(file)

# Class name aliases for backward compatibility
EffNet = DeepFakeDetector
Ensemble = ModelEnsemble

# Entry point
if __name__ == '__main__':
    # Silence deprecation warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    main()
