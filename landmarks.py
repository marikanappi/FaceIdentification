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

def landmarks_dist(png, raw, depth_shape=(480, 640), visualize=False):
    # === Caricamento immagini ===
    if png is str:
        image = cv2.imread(png)
        depth_map = np.fromfile(raw, dtype=np.uint16).reshape(depth_shape)
    else:
        image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        depth_map = np.frombuffer(raw, dtype=np.uint16).reshape(depth_shape)

    # === Face Alignment ===
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fa = face_alignment.FaceAlignment('2D', device=device)
    preds = fa.get_landmarks(image)
    if preds is None or len(preds) == 0:
        print("Nessun volto rilevato.")
        return None

    landmark_ids = {
        'ensx': 36, 'exsx': 39, 'se': 27, 'prn': 30, 'alsx': 31, 'sn': 33,
        'cheilion_sx': 48, 'cheilion_dx': 54, 'ls': 51, 'li': 57,
        'gn': 8, 'ensdx': 45, 'alsdx': 35
    }

    opposite_landmarks = {
        'ensx': 'ensdx', 'ensdx': 'ensx',
        'alsx': 'alsdx', 'alsdx': 'alsx',
        'cheilion_sx': 'cheilion_dx', 'cheilion_dx': 'cheilion_sx',
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
        ("ensx", "se"), ("ensx", "exsx"), ("se", "prn"), ("se", "alsx"),
        ("prn", "alsx"), ("ensx", "alsx"), ("prn", "sn"), ("alsx", "sn"),
        ("ensx", "prn"), ("alsx", "exsx"), ("alsx", "cheilion_sx"),
        ("ensx", "cheilion_sx"), ("prn", "cheilion_sx"), ("se", "gn"),
        ("ls", "li"), ("se", "ls"), ("ensx", "ensdx"), ("alsx", "alsdx"),
        ("cheilion_sx", "cheilion_dx")
    ]

    distances = {}
    for a, b in distance_pairs:
        if a in landmarks_3d and b in landmarks_3d:
            d = dist(landmarks_3d[a], landmarks_3d[b])
            distances[f"dist_{a}_{b}"] = round(d / ref_dist, 4)
        else:
            distances[f"dist_{a}_{b}"] = None

    # === Aree ===
    area_triplets = [
        ("ensx", "exsx", "ensdx"),
        ("prn", "alsx", "alsdx"),
        ("cheilion_sx", "cheilion_dx", "ls"),
        ("ensx", "ensdx", "prn"),
        ("se", "prn", "alsx"),
        ("se", "prn", "alsdx"),
    ]

    areas = {}
    for a, b, c in area_triplets:
        if all(k in landmarks_3d for k in [a, b, c]):
            val = area_triangle(landmarks_3d[a], landmarks_3d[b], landmarks_3d[c])
            areas[f"area_{a}_{b}_{c}"] = round(val / ref_area, 4)
        else:
            areas[f"area_{a}_{b}_{c}"] = None

    # === Volumi ===
    volume_quads = [
        ("ensx", "prn", "alsx", "alsdx"),
        ("ensx", "exsx", "ensdx", "prn"),
        ("ensx", "ensdx", "se", "prn"),
        ("exsx", "ensdx", "alsx", "alsdx")
    ]

    volumes = {}
    for a, b, c, d in volume_quads:
        if all(k in landmarks_3d for k in [a, b, c, d]):
            val = volume_tetra(landmarks_3d[a], landmarks_3d[b], landmarks_3d[c], landmarks_3d[d])
            volumes[f"volume_{a}_{b}_{c}_{d}"] = round(val / ref_volume, 4)
        else:
            volumes[f"volume_{a}_{b}_{c}_{d}"] = None

    # Visualizza landmarks
    if visualize:
        img_copy = image.copy()
        for name, (x, y, _) in landmarks_3d.items():
            cv2.circle(img_copy, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(img_copy, name, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imshow("Landmarks", img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "distances": distances,
        "areas": areas,
        "volumes": volumes,
        "facial_area": round(sum(v for v in areas.values() if v is not None), 4)
    }

'''results = landmarks_dist("ciccia1_Color.png", "ciccia1_Depth.raw", visualize=True)
print("Results:", results)'''