import cv2
import numpy as np
import face_alignment
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import csv
from skimage.feature import local_binary_pattern
from scipy.stats import skew, entropy
import math

import os, csv, sqlite3
import cv2, torch
import numpy as np
import face_alignment
from PIL import Image
from skimage.feature import local_binary_pattern
from scipy.stats import skew, entropy

def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def area_triangle(a, b, c):
    ab = np.array(b) - np.array(a)
    ac = np.array(c) - np.array(a)
    cross = np.cross(ab, ac)
    return np.linalg.norm(cross) / 2

def volume_tetra(a, b, c, d):
    return abs(np.dot((np.array(a) - np.array(d)), 
                      np.cross(np.array(b) - np.array(d), 
                               np.array(c) - np.array(d)))) / 6

def show_overlay(rgb_path, depth_path, width, height, depth_dtype='uint16'):

    rgb = np.array(Image.open(rgb_path).convert("RGB"))

    dtype = np.uint16 if depth_dtype == 'uint16' else np.float32
    depth = np.fromfile(depth_path, dtype=dtype).reshape((height, width))
    
    depth_normalized = (depth - np.min(depth)) / (np.max(depth) - np.min(depth) + 1e-8)
    
    depth_colormap = plt.cm.plasma(depth_normalized)[:, :, :3]

    overlay = (0.5 * rgb / 255 + 0.5 * depth_colormap)

    plt.figure(figsize=(10, 5))
    plt.imshow(overlay)
    plt.title("Overlay RGB + Depth")
    plt.axis('off')
    plt.show()

def calculate_texture_features(image, landmarks_3d, normalize=True):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if not landmarks_3d:
        return None

    x_coords = [v[0] for v in landmarks_3d.values()]
    y_coords = [v[1] for v in landmarks_3d.values()]
    min_x, max_x = max(0, min(x_coords)), min(image.shape[1], max(x_coords))
    min_y, max_y = max(0, min(y_coords)), min(image.shape[0], max(y_coords))
    
    face_region = image[min_y:max_y, min_x:max_x]
    if face_region.size == 0:
        return None

    r = face_region[:,:,0].flatten()
    g = face_region[:,:,1].flatten()
    b = face_region[:,:,2].flatten()

    hsv_region = hsv[min_y:max_y, min_x:max_x]
    h = hsv_region[:,:,0].flatten()
    s = hsv_region[:,:,1].flatten()
    v = hsv_region[:,:,2].flatten()

    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)
    
    features = {
        'mean_R': np.mean(r),
        'mean_G': np.mean(g),
        'mean_B': np.mean(b),
        'std_R': np.std(r),
        'std_G': np.std(g),
        'std_B': np.std(b),
        'skew_R': skew(r),
        'skew_G': skew(g),
        'skew_B': skew(b),
        'entropy_R': entropy(np.histogram(r, bins=256, range=(0,255))[0]),
        'entropy_G': entropy(np.histogram(g, bins=256, range=(0,255))[0]),
        'entropy_B': entropy(np.histogram(b, bins=256, range=(0,255))[0]),
        'mean_H': np.mean(h),
        'mean_S': np.mean(s),
        'mean_V': np.mean(v),
        'lbp_mean': np.mean(lbp),
        'lbp_std': np.std(lbp),
        'lbp_entropy': entropy(lbp_hist)
    }
    
    if normalize:

        color_features = ['mean_R', 'mean_G', 'mean_B', 'std_R', 'std_G', 'std_B']
        for f in color_features:
            features[f] = features[f] / 255.0

        features['mean_H'] = features['mean_H'] / 179.0
        features['mean_S'] = features['mean_S'] / 255.0
        features['mean_V'] = features['mean_V'] / 255.0

        max_lbp = n_points + 2
        features['lbp_mean'] = features['lbp_mean'] / max_lbp
        features['lbp_std'] = features['lbp_std'] / max_lbp
    
    return features

def landmarks_dist(png, raw, identity, expression, depth_shape=(480, 640), visualize=False):
    # === Caricamento immagini ===
    image = cv2.imread(png)
    depth_map = np.fromfile(raw, dtype=np.uint16).reshape(depth_shape)

    # === Face Alignment ===
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fa = face_alignment.FaceAlignment('2D', device=device)
    preds = fa.get_landmarks(image)
    if preds is None or len(preds) == 0:
        print("Nessun volto rilevato.")
        return None

    landmark_ids = {
        'ensx': 36, 
        'exsx': 39, 
        'se': 27, 
        'prn': 30, 
        'alsx': 31, 
        'sn': 33,
        'cheilion_sx': 48, 
        'cheilion_dx': 54, 
        'ls': 51, 
        'li': 57,
        'gn': 8, 
        'ensdx': 45, 
        'alsdx': 35,
        'exdx': 42,
    }

    opposite_landmarks = {
        'ensx': 'ensdx', 
        'ensdx': 'ensx',
        'alsx': 'alsdx', 
        'alsdx': 'alsx',
        'cheilion_sx': 'cheilion_dx', 
        'cheilion_dx': 'cheilion_sx',
    }

    landmarks_3d = {}
    face_landmarks = preds[0]

    for name, idx in landmark_ids.items():
        x, y = map(int, face_landmarks[idx])
        z_depth = 0
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            z_depth = int(depth_map[y, x])
        landmarks_3d[name] = (x, y, z_depth)

    for name, (x, y, z) in landmarks_3d.items():
        if z == 0 and name in opposite_landmarks:
            opp = opposite_landmarks[name]
            if opp in landmarks_3d and landmarks_3d[opp][2] != 0:
                landmarks_3d[name] = (x, y, landmarks_3d[opp][2])

    # === Distanza, area e volume di riferimento ===
    if all(k in landmarks_3d for k in ["ensx", "ensdx"]):
        ref_dist = dist(landmarks_3d["ensx"], landmarks_3d["ensdx"])
    elif all(k in landmarks_3d for k in ["ensx", "alsx"]):
        ref_dist = dist(landmarks_3d["ensx"], landmarks_3d["alsx"])
    else:
        ref_dist = 1

    if all(k in landmarks_3d for k in ["ensx", "ensdx", "prn"]):
        ref_area = area_triangle(landmarks_3d["ensx"], landmarks_3d["ensdx"], landmarks_3d["prn"])
    else:
        ref_area = 1

    if all(k in landmarks_3d for k in ["ensx", "ensdx", "se", "prn"]):
        ref_volume = volume_tetra(landmarks_3d["ensx"], landmarks_3d["ensdx"], landmarks_3d["se"], landmarks_3d["prn"])
    else:
        ref_volume = 1

    # === Distanze ===
    distance_pairs = [
        ("ensx", "se"),        
        ("ensx", "prn"),      
        ("se", "prn"),      
        ("se", "ls"),     
        ("prn", "ls"),      
        ("alsx", "alsdx"),   
        ("cheilion_sx", "cheilion_dx"), 
        ("ensx", "exsx"),  
        ("ensdx", "exdx"),   
        ("exsx", "exdx"),   
        ("ensx", "ensdx"),   
    ]

    distances = []
    for a, b in distance_pairs:
        if a in landmarks_3d and b in landmarks_3d:
            d = dist(landmarks_3d[a], landmarks_3d[b])
            distances.append(round(d / ref_dist, 4))
        else:
            distances.append(0)

    # === Aree ===
    area_triplets = [
        ("ensx", "exsx", "exdx"),   
        ("prn", "alsx", "alsdx"),    
        ("cheilion_sx", "cheilion_dx", "ls"),  
        ("ensx", "ensdx", "prn"), 
        ("se", "prn", "alsx"),  
        ("se", "prn", "alsdx"),     
    ]

    areas = []
    for a, b, c in area_triplets:
        if all(k in landmarks_3d for k in [a, b, c]):
            val = area_triangle(landmarks_3d[a], landmarks_3d[b], landmarks_3d[c])
            areas.append(round(val / ref_area, 4))
        else:
            areas.append(0)

    # === Volumi ===
    volume_quads = [
        ("ensx", "prn", "alsx", "alsdx"),   
        ("ensx", "exsx", "exdx", "prn"),    
        ("ensx", "ensdx", "se", "prn"),      
        ("exsx", "exdx", "alsx", "alsdx")  
    ]

    volumes = []
    for a, b, c, d in volume_quads:
        if all(k in landmarks_3d for k in [a, b, c, d]):
            val = volume_tetra(landmarks_3d[a], landmarks_3d[b], landmarks_3d[c], landmarks_3d[d])
            volumes.append(round(val / ref_volume, 4))
        else:
            volumes.append(0)

    facial_area = float(sum(v for v in areas if v is not None)) if any(areas) else 0

    texture_features = calculate_texture_features(image, landmarks_3d)
    if texture_features is None:
        texture_features = {
            'mean_R': 0, 'mean_G': 0, 'mean_B': 0,
            'std_R': 0, 'std_G': 0, 'std_B': 0,
            'skew_R': 0, 'skew_G': 0, 'skew_B': 0,
            'entropy_R': 0, 'entropy_G': 0, 'entropy_B': 0,
            'mean_H': 0, 'mean_S': 0, 'mean_V': 0,
            'lbp_mean': 0, 'lbp_std': 0, 'lbp_entropy': 0
        }

    feature_vector = [
        *distances,  # 11 distance features
        *areas,      # 6 area features
        *volumes,    # 4 volume features
        facial_area,
        texture_features['mean_R'],
        texture_features['mean_G'],
        texture_features['mean_B'],
        texture_features['std_R'],
        texture_features['std_G'],
        texture_features['std_B'],
        texture_features['skew_R'],
        texture_features['skew_G'],
        texture_features['skew_B'],
        texture_features['entropy_R'],
        texture_features['entropy_G'],
        texture_features['entropy_B'],
        texture_features['mean_H'],
        texture_features['mean_S'],
        texture_features['mean_V'],
        texture_features['lbp_mean'],
        texture_features['lbp_std'],
        texture_features['lbp_entropy']
    ]

    return feature_vector

def process_folder_to_csv(folder_path, output_csv):

    file_exists = os.path.exists(output_csv)

    headers = [
        # Distances (11)
        'dist_ensx_se', 'dist_ensx_prn', 'dist_se_prn', 'dist_se_ls', 'dist_prn_ls',
        'dist_alL_alR', 'dist_cheilion_sx_cheilion_dx', 'dist_ensx_exsx', 
        'dist_ensdx_exdx', 'dist_exsx_exdx', 'dist_ensx_ensdx',
        
        # Areas (6)
        'area_ensx_exsx_exdx', 'area_prn_alL_alR', 'area_cheilion_sx_cheilion_dx_ls',
        'area_ensx_ensdx_prn', 'area_se_prn_alL', 'area_se_prn_alR',
        
        # Volumes (4)
        'volume_ensx_prn_alL_alR', 'volume_ensx_exsx_exdx_prn',
        'volume_ensx_ensdx_se_prn', 'volume_exsx_exdx_alL_alR',
        
        # Facial area
        'facial_area',
        
        # Texture features (18)
        'texture_mean_R', 'texture_mean_G', 'texture_mean_B',
        'texture_std_R', 'texture_std_G', 'texture_std_B',
        'texture_skew_R', 'texture_skew_G', 'texture_skew_B',
        'texture_entropy_R', 'texture_entropy_G', 'texture_entropy_B',
        'texture_mean_H', 'texture_mean_S', 'texture_mean_V',
        'texture_lbp_mean', 'texture_lbp_std', 'texture_lbp_entropy',
        
        # Metadati
        'identity', 'expression', 'filename'
    ]

    rgb_files = [f for f in os.listdir(folder_path) if f.endswith('_Color.png')]
    
    with open(output_csv, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)

        if not file_exists:
            writer.writerow(headers)
        
        for rgb_file in rgb_files:

            base_name = rgb_file.replace('_Color.png', '')
            parts = base_name.split('_')
            identity = parts[0] if len(parts) > 0 else 'unknown'
            expression = parts[1] if len(parts) > 1 else 'neutral'

            rgb_path = os.path.join(folder_path, rgb_file)
            raw_path = os.path.join(folder_path, f"{base_name}_Depth.raw")
            
            if not os.path.exists(raw_path):
                print(f"File di profondità mancante per {rgb_file}, saltando...")
                continue

            try:
                features = landmarks_dist(rgb_path, raw_path, identity, expression)
                if features is None:
                    print(f"Nessun volto rilevato in {rgb_file}, saltando...")
                    continue

                full_features =  [identity, expression, rgb_file] + features

                writer.writerow(full_features)
                print(f"Processato con successo: {rgb_file}")
                
            except Exception as e:
                print(f"Errore durante l'elaborazione di {rgb_file}: {str(e)}")

