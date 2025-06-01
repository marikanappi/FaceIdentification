import cv2
import numpy as np
import face_alignment
import pandas as pd
import os
import torch

# === CONFIGURAZIONE ===
COLOR_IMAGE_PATH = "vale4_Color.png"
DEPTH_RAW_PATH = "vale4_Depth.raw"
DEPTH_SHAPE = (480, 640)  # Adatta se necessario
CSV_INPUT_PATH = "input.csv"   # Percorso CSV input
CSV_OUTPUT_PATH = "output.csv" # Percorso CSV output

# === FUNZIONE DISTANZA ===
def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# === CARICAMENTO IMMAGINI ===
image = cv2.imread(COLOR_IMAGE_PATH)
depth_map = np.fromfile(DEPTH_RAW_PATH, dtype=np.uint16).reshape(DEPTH_SHAPE)

cv2.imshow("RGB Image", image)
cv2.imshow("Depth Map", depth_map.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

# === FACE ALIGNMENT NETWORK ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fa = face_alignment.FaceAlignment('2D', device=device)
preds = fa.get_landmarks(image)

if preds is None or len(preds) == 0:
    print("Nessun volto rilevato.")
    exit()

# === LANDMARK SELEZIONATI (INDICI FAN) ===
landmark_ids = {
    'ensx': 36, 'exsx': 39, 'se': 27, 'prn': 30, 'alsx': 31, 'sn': 33,
    'cheilion_sx': 48, 'cheilion_dx': 54, 'ls': 51, 'li': 57,
    'gn': 8, 'ensdx': 45, 'alsdx': 35
}

opposite_landmarks = {
    'ensx': 'ensdx',
    'ensdx': 'ensx',
    'alsx': 'alsdx',
    'alsdx': 'alsx',
    'cheilion_sx': 'cheilion_dx',
    'cheilion_dx': 'cheilion_sx',
    # Altri opposti se serve
}

landmarks_3d = {}
face_landmarks = preds[0]  # Assume 1 volto

for name, idx in landmark_ids.items():
    x, y = face_landmarks[idx]
    x, y = int(x), int(y)
    z_depth = 0
    if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
        z_depth = int(depth_map[y, x])
    landmarks_3d[name] = (x, y, z_depth)

for name, (x, y, z) in landmarks_3d.items():
    if z == 0 and name in opposite_landmarks:
        opp_name = opposite_landmarks[name]
        if opp_name in landmarks_3d:
            opp_z = landmarks_3d[opp_name][2]
            if opp_z != 0:
                landmarks_3d[name] = (x, y, opp_z)

for name, (x, y, z) in landmarks_3d.items():
    print(f"{name}: (x={x}, y={y}, z={z})")

print("\n📐 Distanze Euclidee 3D:")
pairs = [
    ("ensx", "se"), ("ensx", "exsx"), ("se", "prn"), ("se", "alsx"),
    ("prn", "alsx"), ("ensx", "alsx"), ("prn", "sn"), ("alsx", "sn"),
    ("ensx", "prn"), ("alsx", "exsx"), ("alsx", "cheilion_sx"),
    ("ensx", "cheilion_sx"), ("prn", "ensx"), ("prn", "cheilion_sx"),
    ("se", "gn"), ("ls", "li"), ("se", "ls"), ("ensx", "ensdx"),
    ("alsx", "alsdx"), ("cheilion_sx", "cheilion_dx")
]

for a, b in pairs:
    if a in landmarks_3d and b in landmarks_3d:
        d = dist(landmarks_3d[a], landmarks_3d[b])
        print(f"{a}-{b}: {d:.2f}")
    else:
        print(f"{a}-{b}: dati mancanti")

img_copy = image.copy()
for name, (x, y, _) in landmarks_3d.items():
    cv2.circle(img_copy, (x, y), 4, (0, 255, 0), -1)
    cv2.putText(img_copy, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

cv2.imshow("Landmarks", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''# === ELABORAZIONE CSV ===
if not os.path.exists(CSV_INPUT_PATH):
    print(f"⚠️ CSV non trovato: {CSV_INPUT_PATH}")
    exit()

df = pd.read_csv(CSV_INPUT_PATH)

distance_rows = []
for _, row in df.iterrows():
    row_distances = {"identity": row.get("identity", "unknown"), "expression": row.get("expression", "unknown")}
    landmarks_row = {}

    # Leggi i landmark dal CSV
    for name in ["ensx", "exsx", "se", "prn", "alsx", "sn", "ls", "cheilion_sx", "gn", "li", "cheilion_dx", "ensdx", "alsdx"]:
        try:
            x = row[f"{name}_x"]
            y = row[f"{name}_y"]
            z = row[f"{name}_z"]
            landmarks_row[name] = [x, y, z]
        except KeyError:
            continue

    # Sostituisci z=0 con lo z opposto (se possibile)
    for name, coords in landmarks_row.items():
        x, y, z = coords
        if z == 0 and name in opposite_landmarks:
            opp_name = opposite_landmarks[name]
            if opp_name in landmarks_row:
                opp_z = landmarks_row[opp_name][2]
                if opp_z != 0:
                    landmarks_row[name][2] = opp_z

    # Calcola la distanza di riferimento per la normalizzazione
    reference_dist = None
    if "ensx" in landmarks_row and "alsx" in landmarks_row:
        reference_dist = dist(landmarks_row["ensx"], landmarks_row["alsx"])

    for a, b in pairs:
        norm_col = f"{a}_{b}"
        if a in landmarks_row and b in landmarks_row:
            d = dist(landmarks_row[a], landmarks_row[b])
            if reference_dist and reference_dist > 0:
                row_distances[norm_col] = round(d / reference_dist, 4)
            else:
                row_distances[norm_col] = np.nan
        else:
            row_distances[norm_col] = np.nan

    distance_rows.append(row_distances)

df_dist = pd.DataFrame(distance_rows)
df_dist.to_csv(CSV_OUTPUT_PATH, index=False)
print(f"✅ Distanze normalizzate salvate in: {CSV_OUTPUT_PATH}")
'''