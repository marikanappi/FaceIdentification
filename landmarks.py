import cv2
import numpy as np
import face_alignment
import torch

def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def landmarks_dist(png, raw, depth_shape=(480, 640), visualize=False):
    # === Caricamento immagini ===
    image = cv2.imread(png)
    depth_map = np.fromfile(raw, dtype=np.uint16).reshape(depth_shape)

    # === Face Alignment Network ===
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fa = face_alignment.FaceAlignment('2D', device=device)
    preds = fa.get_landmarks(image)

    if preds is None or len(preds) == 0:
        print("Nessun volto rilevato.")
        return None

    # === Landmark selezionati ===
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
    face_landmarks = preds[0]  # Assume 1 volto

    for name, idx in landmark_ids.items():
        x, y = map(int, face_landmarks[idx])
        z_depth = 0
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            z_depth = int(depth_map[y, x])
        landmarks_3d[name] = (x, y, z_depth)

    # Sostituzione con profondità opposta se 0
    for name, (x, y, z) in landmarks_3d.items():
        if z == 0 and name in opposite_landmarks:
            opp_name = opposite_landmarks[name]
            if opp_name in landmarks_3d:
                opp_z = landmarks_3d[opp_name][2]
                if opp_z != 0:
                    landmarks_3d[name] = (x, y, opp_z)

    # Coppie da calcolare
    pairs = [
        ("ensx", "se"), ("ensx", "exsx"), ("se", "prn"), ("se", "alsx"),
        ("prn", "alsx"), ("ensx", "alsx"), ("prn", "sn"), ("alsx", "sn"),
        ("ensx", "prn"), ("alsx", "exsx"), ("alsx", "cheilion_sx"),
        ("ensx", "cheilion_sx"), ("prn", "ensx"), ("prn", "cheilion_sx"),
        ("se", "gn"), ("ls", "li"), ("se", "ls"), ("ensx", "ensdx"),
        ("alsx", "alsdx"), ("cheilion_sx", "cheilion_dx")
    ]

    distances = {}
    for a, b in pairs:
        if a in landmarks_3d and b in landmarks_3d:
            d = dist(landmarks_3d[a], landmarks_3d[b])
            distances[f"{a}-{b}"] = round(d, 2)
        else:
            distances[f"{a}-{b}"] = None

    # Visualizzazione opzionale
    if visualize:
        img_copy = image.copy()
        for name, (x, y, _) in landmarks_3d.items():
            cv2.circle(img_copy, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(img_copy, name, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imshow("Landmarks", img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return distances
