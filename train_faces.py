import face_recognition
import pickle
import os

# Konfigurasi Path
DATASET_PATH = "dataset_siswa"
ENCODINGS_PATH = "face_encodings.pkl"

def encode_known_faces():
    known_encodings = []
    known_names = []

    # Cek apakah folder dataset ada
    if not os.path.exists(DATASET_PATH):
        print(f"Folder '{DATASET_PATH}' tidak ditemukan. Buat folder dan isi foto siswa dulu.")
        return

    print("[INFO] Memulai proses encoding wajah siswa...")
    
    # Loop semua file di folder dataset
    for filename in os.listdir(DATASET_PATH):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(filename)[0] 
            
            try:
                name_part, nisn_part = name.rsplit('_', 1)
                
                # Opsional: Rapikan nama (ganti _ jadi spasi)
                display_name = name_part.replace('_', ' ')
                
                # KITA SIMPAN NISN SEBAGAI KUNCI UTAMA DI MEMORY
                # Karena Database SQL mencari berdasarkan NISN/NIM
                identity = nisn_part 
                
                print(f"Processing: {display_name} (NISN: {identity})")
                
            except ValueError:
                # Jika format salah (misal cuma "budi.jpg" tanpa NIM)
                print(f"[WARNING] Format file salah: {filename}. Gunakan format Nama_NISN.jpg")
                continue

            image_path = os.path.join(DATASET_PATH, filename)
            
            # Load gambar
            image = face_recognition.load_image_file(image_path)
            
            try:
                encodings = face_recognition.face_encodings(image)
                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(identity)
                    print(f"✅ Berhasil: {identity}")
                else:
                    print(f"⚠️ Wajah tidak terdeteksi: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Simpan data encoding ke file pickle
    print("[INFO] Menyimpan data ke file...")
    data = {"encodings": known_encodings, "names": known_names}
    with open(ENCODINGS_PATH, "wb") as f:
        f.write(pickle.dumps(data))
    
    print(f"[INFO] Selesai! {len(known_names)} wajah siswa telah tersimpan.")

if __name__ == "__main__":
    encode_known_faces()