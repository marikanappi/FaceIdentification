import cv2
import numpy as np
import face_alignment
import torch

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

    # Correzione z se nullo
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
        ("ensx", "exsx"),
        ("se", "prn"),
        ("ensx", "prn"),
        ("se", "ls"),
        ("ensx", "ensdx"),
        ("alsx", "alsdx"),
        ("cheilion_sx", "cheilion_dx"),
        ("prn", "ls"),
        ("ensdx", "exdx"),
        ("exsx", "exdx"),
    ]

    distances = []
    for a, b in distance_pairs:
        if a in landmarks_3d and b in landmarks_3d:
            d = dist(landmarks_3d[a], landmarks_3d[b])
            distances.append(round(d / ref_dist, 4))
        else:
            distances.append(None)

    # === Aree ===
    area_triplets = [
        ("ensx", "exsx", "ensdx"),
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
            areas.append(None)

    # === Volumi ===
    volume_quads = [
        ("ensx", "prn", "alsx", "alsdx"),
        ("ensx", "exsx", "ensdx", "prn"),
        ("ensx", "ensdx", "se", "prn"),
        ("exsx", "ensdx", "alsx", "alsdx")
    ]

    volumes = []
    for a, b, c, d in volume_quads:
        if all(k in landmarks_3d for k in [a, b, c, d]):
            val = volume_tetra(landmarks_3d[a], landmarks_3d[b], landmarks_3d[c], landmarks_3d[d])
            volumes.append(round(val / ref_volume, 4))
        else:
            volumes.append(None)

    facial_area = float(sum(v for v in areas if v is not None)) if any(areas) else None
    # Restituisce solo le liste di distanze, aree e volumi
    feature_vector = []
    feature_vector.extend(distances)
    feature_vector.extend(areas)
    feature_vector.extend(volumes)
    feature_vector.append(facial_area)

    # Sostituisci None con 0 (o meglio: media del training set, se hai salvato)
    feature_vector = [0 if v is None else v for v in feature_vector]

    return feature_vector

'''feature_vector = landmarks_dist("ciccia1_Color.png", "ciccia1_Depth.raw", "ciccia1", "neutral")
print("Feature Vector:", feature_vector)'''