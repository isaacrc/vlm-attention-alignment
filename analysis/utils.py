import os
import matplotlib.pyplot as plt
import seaborn as sns
#from scipy.misc import imread, imresize
from imageio import imread
from skimage.transform import resize
import torch
import json
from PIL import Image
import skimage.transform
import numpy as np
from scipy.spatial.distance import squareform
import random
import pandas as pd
from tqdm import tqdm
from scipy import stats
from itertools import combinations
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gc
from tqdm import tqdm
from itertools import combinations
import cv2
from scipy.stats import spearmanr

# --- Functions ---
def normalize_attention(attention_map):
    """Normalize attention map to 0-1 range"""
    return (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-10)


def load_human_subject_data(directory):
    """
    Loads all subject heatmap data from a given directory.
    Returns a dictionary: {subject_id: {image_name: 336x336_array}}
    """
    all_subject_data = {}
    #print(f"Loading human subject data from: {directory}")
    
    # Iterate through .npy files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('_heatmaps.npy'):
            subject_id = filename.split('_heatmaps.npy')[0] # Extracts 'subXXX'
            filepath = os.path.join(directory, filename)
            try:
                # Load the dictionary for this subject
                # Use allow_pickle=True because you are loading a dict saved by np.save
                subject_heatmaps = np.load(filepath, allow_pickle=True).item()
                all_subject_data[subject_id] = subject_heatmaps
            except Exception as e:
                print(f"  Error loading {filepath}: {e}")
    #print(f"Loaded data for {len(all_subject_data)} subjects from {directory}")
    return all_subject_data

def average_layers_all_images(heatmap_dict):
    """
    Iterates over all images in the dictionary and averages all layers.
    Returns: {image_id: averaged_attention_map}
    """
    averaged = {}

    for image_id, layers_dict in heatmap_dict.items():
        # Extract all layer heatmaps for this image
        layer_maps = [layers_dict[layer] for layer in layers_dict.keys()]

        # Average across layers
        avg_map = np.mean(np.stack(layer_maps), axis=0)

        averaged[image_id] = avg_map

    return averaged