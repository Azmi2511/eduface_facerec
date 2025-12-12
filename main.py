from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import face_recognition
import pickle
import numpy as np
import cv2
import os
import shutil
from datetime import datetime, timedelta
from typing import List

# Import Firebase & MySQL
import firebase_admin
from firebase_admin import credentials, messaging
import mysql.connector
from mysql.connector import Error

# --- KONFIGURASI DATABASE ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'atlas'
}

def get_db_connection():
    """Membuat koneksi ke MySQL"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"[ERROR SQL] Gagal koneksi: {e}")
        return None

# --- KONFIGURASI FIREBASE ---
FIREBASE_ACTIVE = False
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    print("[INFO] Firebase berhasil terhubung!")
    FIREBASE_ACTIVE = True
except Exception as e:
    print(f"[WARNING] Firebase gagal load: {e}")
    print("Fitur notifikasi tidak akan berjalan, tapi absensi tetap bisa.")

# --- GLOBAL VARIABLES ---
last_attendance_time = {}
COOLDOWN_PERIOD = timedelta(minutes=60) # Siswa tidak bisa absen lagi dalam 60 menit

DATASET_PATH = "dataset_siswa"
ENCODINGS_PATH = "face_encodings.pkl"

if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

# Database Memory (Pickle)
db_face = {"encodings": [], "names": [], "parent_tokens": {}}

# --- INIT APP ---
app = FastAPI(title="ATLAS Face Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HELPER FUNCTIONS ---

def load_database():
    global db_face
    try:
        with open(ENCODINGS_PATH, "rb") as f:
            loaded = pickle.loads(f.read())
            if "parent_tokens" not in loaded:
                loaded["parent_tokens"] = {}
            db_face = loaded
        print(f"[INFO] Pickle dimuat: {len(db_face['names'])} siswa.")
    except FileNotFoundError:
        print("[INFO] Pickle belum ada, memulai dari kosong.")
        db_face = {"encodings": [], "names": [], "parent_tokens": {}}

def save_database():
    with open(ENCODINGS_PATH, "wb") as f:
        f.write(pickle.dumps(db_face))

def validate_face_quality(rgb_img, face_locations):
    """Memastikan foto layak untuk pendaftaran mandiri"""
    if len(face_locations) == 0:
        return False, "Wajah tidak terdeteksi."
    
    if len(face_locations) > 1:
        return False, "Terdeteksi lebih dari 1 wajah. Harap foto sendirian."

    # Cek Ukuran Wajah (Apakah terlalu jauh?)
    top, right, bottom, left = face_locations[0]
    face_width = right - left
    img_width = rgb_img.shape[1]
    
    if (face_width / img_width) < 0.15: 
        return False, "Wajah terlalu jauh/kecil. Mohon dekatkan kamera."

    # Cek Kecerahan
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    brightness = np.mean(hsv[:,:,2])
    
    if brightness < 60: return False, "Foto terlalu gelap."
    if brightness > 230: return False, "Foto terlalu silau."

    return True, "OK"

def send_firebase_notif(token, student_name, time_str, status_absen):
    """Mengirim notifikasi ke HP Orang Tua"""
    if not FIREBASE_ACTIVE or not token:
        return

    try:
        message = messaging.Message(
            notification=messaging.Notification(
                title="Laporan Kehadiran ATLAS",
                body=f"Ananda {student_name} tercatat {status_absen} pada pukul {time_str}."
            ),
            token=token
        )
        response = messaging.send(message)
        print(f"[NOTIF] Terkirim ke {student_name}: {response}")
    except Exception as e:
        print(f"[ERROR NOTIF] {e}")

# Load DB saat start
load_database()

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
def index():
    # Pastikan file index.html atau daftar.html ada
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return "<h1>ATLAS API Running</h1>"

@app.get("/students", response_model=List[str])
def get_all_students():
    return db_face["names"]

@app.post("/register")
async def register_student(
    name: str = Form(...), 
    nisn: str = Form(...),
    pose: str = Form(...), # <--- PARAMETER BARU (depan/kiri/kanan)
    file: UploadFile = File(...)
):
    # 1. Bersihkan Input
    clean_nisn = str(nisn).strip()
    clean_name = name.strip().replace(" ", "_")
    
    # Nama file jadi unik: Budi_12345_depan.jpg
    filename = f"{clean_name}_{clean_nisn}_{pose}.jpg" 

    print(f"[REGISTER] Mendaftarkan {clean_name} ({clean_nisn}) - Pose: {pose}")

    # 2. Cek Koneksi & Data Master
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        cursor = conn.cursor(dictionary=True)
        # Cek apakah siswa ada di database sekolah
        cursor.execute("SELECT * FROM students WHERE nisn = %s", (clean_nisn,))
        student = cursor.fetchone()
        
        if not student:
            conn.close()
            raise HTTPException(status_code=404, detail=f"NISN {clean_nisn} tidak terdaftar.")
        
        # 3. Proses Gambar & Validasi
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_img)
        
        # Validasi Kualitas (Opsional: Bisa dilonggarkan untuk pose samping)
        is_valid, msg = validate_face_quality(rgb_img, face_locations)
        if not is_valid:
            conn.close()
            raise HTTPException(status_code=400, detail=f"Pose {pose}: {msg}")
            
        new_encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]

        # 4. Simpan File Fisik
        with open(os.path.join(DATASET_PATH, filename), "wb") as f:
            f.write(contents)

        # 5. UPDATE MEMORY PICKLE
        # --- PERUBAHAN UTAMA DI SINI ---
        # KITA TIDAK LAGI MENGHAPUS DATA LAMA. KITA APPEND.
        
        # Cek jika pose ini sudah ada sebelumnya, baru hapus yang spesifik itu (opsional)
        # Tapi biar simple, kita append saja. Face Recognition bisa handle multiple encodings per nama.
        
        db_face["encodings"].append(new_encoding)
        db_face["names"].append(clean_nisn) # Nama di memory tetap NISN
        save_database()

        # 6. Update MySQL (Tandai sudah registrasi)
        # Kita update photo_path dengan foto terakhir yang diupload
        sql_update = """
            UPDATE students 
            SET is_face_registered = 1, 
                photo_path = %s,
                face_registered_at = NOW()
            WHERE nisn = %s
        """
        cursor.execute(sql_update, (filename, clean_nisn))
        conn.commit()
        conn.close()
        
        return {"status": "success", "message": f"Foto pose '{pose}' berhasil disimpan."}

    except Exception as e:
        if conn.is_connected(): conn.close()
        print(f"[ERROR REGISTER] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_face(file: UploadFile = File(...)):
    # 1. Decode Gambar
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Deteksi Wajah
    face_locations = face_recognition.face_locations(rgb_img, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    processed_results = []
    conn = get_db_connection()

    if len(db_face["names"]) == 0:
         return {"status": "error", "message": "Database wajah kosong."}

    current_time_obj = datetime.now()
    current_time_str = current_time_obj.strftime('%H:%M:%S')
    current_date_str = current_time_obj.strftime('%Y-%m-%d')

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(db_face["encodings"], encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(db_face["encodings"], encoding)
        
        detected_nisn = None
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            detected_nisn = db_face["names"][best_match_index]
        
        # Default Value
        real_name = "Unknown"
        status = "unknown"

        if detected_nisn:
            if conn:
                try:
                    cursor = conn.cursor(dictionary=True)
                    
                    # Ambil Nama & Token FCM Ortu
                    sql_info = """
                        SELECT s.full_name, p.fcm_token 
                        FROM students s
                        LEFT JOIN parents p ON s.parent_id = p.id
                        WHERE s.nisn = %s
                    """
                    cursor.execute(sql_info, (detected_nisn,))
                    student_data = cursor.fetchone()
                    
                    if student_data:
                        real_name = student_data['full_name']
                        parent_token = student_data['fcm_token']
                        
                        # --- LOGIKA COOLDOWN ---
                        should_record = False
                        if detected_nisn not in last_attendance_time:
                            should_record = True
                        elif current_time_obj - last_attendance_time[detected_nisn] > COOLDOWN_PERIOD:
                            should_record = True
                        
                        if should_record:
                            # --- CEK JAM TERLAMBAT ---
                            # Default jam 7 pagi jika tidak ada setting
                            limit_time_str = "07:00:00"
                            
                            # Coba ambil setting dari DB (jika ada tabel system_settings)
                            try:
                                cursor.execute("SELECT late_limit FROM system_settings LIMIT 1")
                                setting_res = cursor.fetchone()
                                if setting_res and setting_res['late_limit']:
                                    # Handle tipe data timedelta atau string
                                    val = setting_res['late_limit']
                                    if isinstance(val, timedelta):
                                        total_seconds = int(val.total_seconds())
                                        h, m, s = (total_seconds // 3600), (total_seconds % 3600) // 60, total_seconds % 60
                                        limit_time_str = f"{h:02}:{m:02}:{s:02}"
                                    else:
                                        limit_time_str = str(val)
                            except:
                                pass # Gunakan default 07:00:00 jika tabel settings belum ada

                            # Tentukan Status
                            attendance_status = 'Hadir'
                            if current_time_str > limit_time_str:
                                attendance_status = 'Terlambat'

                            # Insert Log
                            sql_insert = """
                                INSERT INTO attendance_logs 
                                (student_nisn, date, time_log, status)
                                VALUES (%s, %s, %s, %s)
                            """
                            cursor.execute(sql_insert, (detected_nisn, current_date_str, current_time_str, attendance_status))
                            conn.commit()
                            
                            last_attendance_time[detected_nisn] = current_time_obj
                            status = "recorded"
                            
                            # --- KIRIM NOTIFIKASI ---
                            if parent_token:
                                send_firebase_notif(parent_token, real_name, current_time_str, attendance_status)
                        
                        else:
                            status = "ignored" # Masih Cooldown
                            
                except Error as e:
                    print(f"[SQL ERROR PREDICT] {e}")
                    real_name = f"NISN:{detected_nisn}"
            else:
                real_name = f"NISN:{detected_nisn} (No DB)"
        
        processed_results.append({
            "nisn": detected_nisn if detected_nisn else "Unknown",
            "name": real_name,
            "status": status,
            "timestamp": current_time_str
        })

    if conn: conn.close()

    # Filter hasil agar response tidak penuh dengan 'Unknown' jika banyak noise
    new_entries = [res for res in processed_results if res['status'] == 'recorded']

    return {
        "status": "success", 
        "new_entries": new_entries,
        "all_detected": processed_results
    }

# Endpoint lain (Update/Delete/History) tetap sama...
class UpdateStudentModel(BaseModel):
    new_name: str

@app.delete("/students/{name}")
def delete_student(name: str):
    clean_name = name.strip().replace(" ", "_") # Ingat nama di memory adalah NISN
    
    if clean_name not in db_face["names"]:
        raise HTTPException(status_code=404, detail="Siswa tidak ditemukan")

    index = db_face["names"].index(clean_name)
    del db_face["names"][index]
    del db_face["encodings"][index]
    save_database()
    
    # Hapus file fisik (Logic perlu disesuaikan karena kita menyimpan Nama_NISN)
    # Untuk simplifikasi, user menghapus manual atau logic pencarian file ditingkatkan
    return {"status": "success", "message": "Data encoding dihapus."}

@app.get("/history")
def get_attendance_history(date: str = None):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="DB Error")
    
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')

    results = []
    try:
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT a.time_log, a.student_nisn, s.full_name, a.status 
            FROM attendance_logs a
            JOIN students s ON a.student_nisn = s.nisn
            WHERE a.date = %s
            ORDER BY a.time_log DESC
        """
        cursor.execute(query, (date,))
        results = cursor.fetchall()
        
        # Konversi tipe data timedelta/time ke string agar JSON valid
        for row in results:
            if isinstance(row['time_log'], (timedelta, datetime)):
                 row['time_log'] = str(row['time_log'])
                 
    except Error as e:
        print(f"[SQL ERROR] {e}")
    finally:
        conn.close()

    return {"date": date, "total": len(results), "data": results}

@app.get("/daftar", response_class=HTMLResponse)
def page_daftar():
    return FileResponse("daftar.html")