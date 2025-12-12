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
import firebase_admin
from firebase_admin import credentials, messaging
import mysql.connector
from mysql.connector import Error

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

try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    print("[INFO] Firebase berhasil terhubung!")
    FIREBASE_ACTIVE = True
except Exception as e:
    print(f"[WARNING] Firebase gagal load: {e}")
    print("Fitur notifikasi tidak akan berjalan, tapi absensi tetap bisa.")
    FIREBASE_ACTIVE = False

last_attendance_time = {}
period = 60
COOLDOWN_PERIOD = timedelta(minutes=period)

app = FastAPI(title="ATLAS Face Recognition API Documentation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASET_PATH = "dataset_siswa"
ENCODINGS_PATH = "face_encodings.pkl"

if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

db_face = {"encodings": [], "names": [], "parent_tokens": {}}


def load_database():
    """Memuat data dari file pickle ke memory saat startup"""
    global db_face
    try:
        with open(ENCODINGS_PATH, "rb") as f:
            loaded = pickle.loads(f.read())
            if "parent_tokens" not in loaded:
                loaded["parent_tokens"] = {}
            db_face = loaded
        print(f"[INFO] Database dimuat: {len(db_face['names'])} siswa terdaftar.")
    except FileNotFoundError:
        print("[INFO] Database belum ada, memulai dari kosong.")
        db_face = {"encodings": [], "names": [], "parent_tokens": {}}

def save_database():
    """Menyimpan data memory ke file pickle"""
    print("[INFO] Menyimpan perubahan ke database...")
    with open(ENCODINGS_PATH, "wb") as f:
        f.write(pickle.dumps(db_face))

load_database()

class UpdateStudentModel(BaseModel):
    new_name: str

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse("index.html")
    # return {
    #     "message": "ATLAS API Ready",
    #     "total_students": len(db_face["names"])
    # }

@app.get("/students", response_model=List[str])
def get_all_students():
    """Mengambil semua nama siswa yang terdaftar"""
    return db_face["names"]

@app.post("/register")
async def register_student(
    name: str = Form(...), 
    nisn: str = Form(...), 
    file: UploadFile = File(...)
):
    clean_nisn = nisn.strip()
    clean_name = name.strip().replace(" ", "_")
    filename = f"{clean_name}_{clean_nisn}.jpg"

    conn = get_db_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM students WHERE nisn = %s", (clean_nisn,))
        student = cursor.fetchone()
        
        if not student:
            conn.close()
            raise HTTPException(status_code=404, detail="NISN tidak ditemukan di Data Master Siswa. Hubungi Admin Sekolah.")
    
    contents = await file.read()
    rgb_img = cv2.cvtColor(cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)
    
    if not face_locations:
        raise HTTPException(status_code=400, detail="Wajah tidak terdeteksi")
        
    new_encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]

    with open(os.path.join(DATASET_PATH, filename), "wb") as f:
        f.write(contents)

    db_face["encodings"].append(new_encoding)
    db_face["names"].append(clean_nisn)
    save_database()

    if conn:
        try:
            cursor = conn.cursor()
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
            print(f"[SQL] Data siswa {clean_nisn} berhasil diupdate.")
        except Error as e:
            print(f"[SQL ERROR] {e}")
    

    return {"status": "success", "message": f"Wajah siswa {name} berhasil didaftarkan."}

@app.post("/predict")
async def predict_face(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_img, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    processed_results = []
    conn = get_db_connection()

    DEVICE_ID = 1

    if len(db_face["names"]) == 0:
         return {"status": "error", "message": "Database kosong."}

    current_time = datetime.now()

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(db_face["encodings"], encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(db_face["encodings"], encoding)
        
        name = "Unknown"
        status = "unknown"
        
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = db_face["names"][best_match_index]
            detected_nisn = db_face["names"][best_match_index]
            real_name = "Unknown"

            if conn:
                try:
                    cursor = conn.cursor(dictionary=True)
                    
                    sql_info = """
                        SELECT s.full_name FROM students s
                        WHERE s.nisn = %s
                    """
                    cursor.execute(sql_info, (detected_nisn,))
                    student_data = cursor.fetchone()
                    
                    if student_data:
                        real_name = student_data['full_name']
                        
                        should_record = False
                        if detected_nisn not in last_attendance_time:
                            should_record = True
                        elif datetime.now() - last_attendance_time[detected_nisn] > COOLDOWN_PERIOD:
                            should_record = True
                        
                        if should_record:
                            current_date = datetime.now().strftime('%Y-%m-%d')
                            current_time = datetime.now().strftime('%H:%M:%S')
                            
                            sql_limit = "SELECT late_limit FROM system_settings LIMIT 1"
                            cursor.execute(sql_limit)
                            result = cursor.fetchone()

                            limit_time_str = "07:00:00"
                            
                            if result and result['late_limit']: 
                                db_time = result['late_limit']
                                
                                if isinstance(db_time, timedelta):
                                    total_seconds = int(db_time.total_seconds())
                                    hours = total_seconds // 3600
                                    minutes = (total_seconds % 3600) // 60
                                    seconds = total_seconds % 60
                                    limit_time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
                                else:
                                    temp_str = str(db_time)
                                    if len(temp_str) == 7:
                                        limit_time_str = "0" + temp_str
                                    else:
                                        limit_time_str = temp_str
                            
                            attendance_status = ""

                            if current_time > limit_time_str:
                                attendance_status = 'Terlambat'
                            else:
                                attendance_status = 'Hadir'

                            sql_insert = """
                                INSERT INTO attendance_logs 
                                (student_nisn, date, time_log, status)
                                VALUES (%s, %s, %s, %s)
                            """
                            cursor.execute(sql_insert, (detected_nisn, current_date, current_time, attendance_status))
                            conn.commit()
                            
                            last_attendance_time[detected_nisn] = datetime.now()
                            status = "recorded"
                        
                        else:
                            status = "ignored"
                            
                except Error as e:
                    print(f"[SQL ERROR] {e}")
                    real_name = f"NISN:{detected_nisn}"
        
        processed_results.append({
            "nisn": name,
            "name": real_name,
            "status": status,
            "timestamp": current_time
        })

    new_entries = [res for res in processed_results if res['status'] == 'recorded']
    
    if conn:
        conn.close()

    return {
        "status": "success", 
        "new_entries": new_entries,
        "all_detected": processed_results
    }

@app.delete("/students/{name}")
def delete_student(name: str):
    """Menghapus data siswa dan fotonya"""
    clean_name = name.strip().replace(" ", "_").lower()

    if clean_name not in db_face["names"]:
        raise HTTPException(status_code=404, detail="Siswa tidak ditemukan")

    index = db_face["names"].index(clean_name)

    del db_face["names"][index]
    del db_face["encodings"][index]

    file_path = os.path.join(DATASET_PATH, f"{clean_name}.jpg")
    if os.path.exists(file_path):
        os.remove(file_path)

    save_database()

    return {"status": "success", "message": f"Data siswa {clean_name} telah dihapus."}

@app.put("/students/{old_name}")
def update_student_name(old_name: str, data: UpdateStudentModel):
    """Mengganti nama siswa (Rename)"""
    clean_old_name = old_name.strip().replace(" ", "_").lower()
    clean_new_name = data.new_name.strip().replace(" ", "_").lower()

    if clean_old_name not in db_face["names"]:
        raise HTTPException(status_code=404, detail="Siswa lama tidak ditemukan")
    
    if clean_new_name in db_face["names"]:
        raise HTTPException(status_code=400, detail="Nama baru sudah dipakai oleh siswa lain")

    index = db_face["names"].index(clean_old_name)

    db_face["names"][index] = clean_new_name

    old_path = os.path.join(DATASET_PATH, f"{clean_old_name}.jpg")
    new_path = os.path.join(DATASET_PATH, f"{clean_new_name}.jpg")
    
    if os.path.exists(old_path):
        os.rename(old_path, new_path)

    save_database()

    return {"status": "success", "message": f"Berhasil mengubah {clean_old_name} menjadi {clean_new_name}"}

class TokenRegistration(BaseModel):
    student_name: str
    fcm_token: str

@app.get("/history")
def get_attendance_history(date: str = None):
    """
    Mengambil data absensi berdasarkan tanggal.
    Format date: YYYY-MM-DD (Misal: 2023-10-27)
    Jika kosong, default ke hari ini.
    """
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')

    results = []
    try:
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT 
                a.time_log, 
                a.student_nisn, 
                s.full_name, 
                a.status 
            FROM attendance_logs a
            JOIN students s ON a.student_nisn = s.nisn
            WHERE a.date = %s
            ORDER BY a.time_log DESC
        """
        cursor.execute(query, (date,))
        results = cursor.fetchall()
        
    except Error as e:
        print(f"[SQL ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

    return {
        "date": date,
        "total": len(results),
        "data": results
    }