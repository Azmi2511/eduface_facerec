from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import pickle
import numpy as np
import cv2
import os

app = FastAPI(title="ATLAS AI Microservice")

# Izinkan akses dari Laravel & Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Konfigurasi Path
DATASET_PATH = "dataset_siswa"
ENCODINGS_PATH = "face_encodings.pkl"

if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

# Load Memory Wajah
db_face = {"encodings": [], "names": []}

def load_database():
    global db_face
    try:
        with open(ENCODINGS_PATH, "rb") as f:
            db_face = pickle.loads(f.read())
        print(f"[INFO] AI Ready. Memuat {len(db_face['names'])} wajah.")
    except FileNotFoundError:
        print("[INFO] Database wajah baru dibuat.")

def save_database():
    with open(ENCODINGS_PATH, "wb") as f:
        f.write(pickle.dumps(db_face))

load_database()

# --- ENDPOINT 1: DAFTARKAN WAJAH (Dipanggil oleh Laravel saat Register) ---
@app.post("/register-face")
async def register_face(
    nisn: str = Form(...),
    pose: str = Form(...), # depan/kiri/kanan
    file: UploadFile = File(...)
):
    # 1. Proses Gambar
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Deteksi Wajah
    face_locations = face_recognition.face_locations(rgb_img)
    if not face_locations:
        raise HTTPException(status_code=400, detail="Wajah tidak terdeteksi AI")
    
    # 3. Encoding
    new_encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]

    # 4. Simpan ke Memory Pickle (Hanya NISN yang disimpan sebagai ID)
    db_face["encodings"].append(new_encoding)
    db_face["names"].append(nisn)
    save_database()

    # 5. Simpan File Backup (Opsional, agar Python punya arsip)
    filename = f"{nisn}_{pose}.jpg"
    with open(os.path.join(DATASET_PATH, filename), "wb") as f:
        f.write(contents)

    return {"status": "success", "message": "Wajah berhasil dipelajari AI"}

# --- ENDPOINT 2: KENALI WAJAH (Dipanggil oleh CCTV JS) ---
@app.post("/scan-face")
async def scan_face(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Deteksi
    face_locations = face_recognition.face_locations(rgb_img, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    detected_nisns = []

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(db_face["encodings"], encoding, tolerance=0.45)
        face_distances = face_recognition.face_distance(db_face["encodings"], encoding)
        
        if len(db_face["names"]) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                found_nisn = db_face["names"][best_match_index]
                detected_nisns.append(found_nisn)

    # Hapus duplikat (misal 1 siswa terdeteksi 2x di frame yg sama)
    unique_nisns = list(set(detected_nisns))
    
    if unique_nisns:
        return {"status": "found", "nisn_list": unique_nisns}
    else:
        return {"status": "unknown", "nisn_list": []}