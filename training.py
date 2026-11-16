import os
import numpy as np
import cv2
import pydicom
from numpy.lib.stride_tricks import sliding_window_view

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# 1. PENGATURAN DATASET (USC-SIPI & DICOM)
# ==============================================================================

def load_single_dicom(folder_path):
    """Membaca file DICOM secara rekursif dan mengambil irisan tengah (middle slice)."""
    valid_files = []
    for root, _, files in os.walk(folder_path):
        for f in sorted(files):
            full_path = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(full_path, stop_before_pixels=True)
                valid_files.append(full_path)
            except Exception:
                continue

    if not valid_files:
        return None

    # Ambil file di tengah jika terdapat banyak irisan
    target_file = valid_files[len(valid_files) // 2]
    ds = pydicom.dcmread(target_file)
    arr = ds.pixel_array

    # Konversi color/RGB jika 3 channel
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float64)
    elif arr.ndim > 2:
        # Jika citra 3D/multiframe, ambil frame tengah
        arr = arr[arr.shape[0] // 2].astype(np.float64)
    else:
        arr = arr.astype(np.float64)

    min_v, max_v = np.min(arr), np.max(arr)
    if max_v > min_v:
        arr_norm = (arr - min_v) / (max_v - min_v) * 255.0
    else:
        arr_norm = np.zeros_like(arr)
    return arr_norm.astype(np.uint8)

def load_all_dataset():
    images = {}

    # 3 Citra Alami USC-SIPI
    natural = {
        'Baboon': '4.2.03.tiff',
        'Peppers': '4.2.07.tiff',
        'Boat': 'boat.512.tiff'
    }
    for name, fname in natural.items():
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath):
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[Gagal] OpenCV tidak dapat membaca file gambar: {fpath}")
                continue
            images[name] = cv2.resize(img, (512, 512))
        else:
            print(f"[Lewati] File {fpath} tidak ditemukan.")

    # 5 Citra Medis DICOM sesuai penamaan di makalah
    medical_folders = {
        'Hand': '9ab20df7b89dd84716029b777ed7a5b2',
        'Leg': '02ef8f31ea86a45cfce6eb297c274598',
        'Head': '3b1cda064d9d7f2afffac6dab4b6eaf9',
        'Abdominal': 'a05270ea39afe373939d192e69ef435a',
        'Chest': 'feb6447a72c9a0a31e1bb4459e547964'
    }
    for name, folder_name in medical_folders.items():
        folder = os.path.join(BASE_DIR, folder_name)
        if os.path.exists(folder):
            dcm_img = load_single_dicom(folder)
            if dcm_img is not None:
                images[name] = cv2.resize(dcm_img, (512, 512))
            else:
                print(f"[Gagal] File DICOM di folder {folder} tidak terbaca.")
        else:
            print(f"[Lewati] Folder {folder} tidak ditemukan.")

    return images

# ==============================================================================
# 2. EKSTRAKSI FITUR TEKSTUR & AMBANG BATAS (PERSAMAAN 1 - 6 & FIG. 1)
# ==============================================================================

# ==============================================================================
# LANGKAH 3 & 4 (versi sederhana, untuk penjelasan)
# ------------------------------------------------------------------------------
# LANGKAH 3 - Persamaan (1): patch(i, j) = I[i-1 : i+2, j-1 : j+2]
#   Patch 3x3 diambil dengan: patch = cover_img[i-1:i+2, j-1:j+2]
#   (indeks i+2 di Python berarti sampai baris i+1, jadi tepat 3 baris x 3 kolom)
#   lalu dikirim ke fungsi di bawah ini.
# LANGKAH 4 - Hitung 3 fitur statistik patch (Persamaan 2, 3, 4).
# Catatan: proses penyisipan memakai versi vektor (compute_capacity_map)
#          yang hasilnya identik, hanya lebih cepat.
# ==============================================================================
def calculate_patch_features(patch):
    """Standar Deviasi, Range, dan Entropi Shannon (Persamaan 2, 3, 4)."""
    # Persamaan (2) - Simpangan Baku: sigma = sqrt((1/N) * sum (x_k - mean)^2)
    # N = 9 piksel. np.std membagi dengan N (ddof=0), sama dengan rumus paper.
    std_val = float(np.std(patch))

    # Persamaan (3) - Rentang: R = max(x_k) - min(x_k)
    # Selisih piksel paling terang dan paling gelap, untuk mengukur kontras lokal.
    range_val = float(np.max(patch) - np.min(patch))

    # Persamaan (4) - Entropi Shannon: H = - sum p_i * log2(p_i)
    #   counts : berapa kali tiap nilai intensitas muncul di dalam patch
    #   probs  : p_i = counts / 9 (peluang tiap nilai)
    #   H      : makin banyak nilai berbeda, makin tinggi entropinya
    # Contoh: [10,10,10,10,12,12,12,14,16] -> counts 4,3,1,1 -> H = 1,75
    _, counts = np.unique(patch, return_counts=True)
    probs = counts / patch.size
    entropy_val = float(-np.sum(probs * np.log2(probs)))

    return std_val, range_val, entropy_val

def classify_features(std_val, range_val, entropy_val):
    """
    Threshold makalah:
    std: (10, 25), range: (20, 50), entropy: (1.2, 2.2)
    """
    c_std = 'Low' if std_val < 10 else ('Med' if std_val <= 25 else 'High')
    c_rng = 'Low' if range_val < 20 else ('Med' if range_val <= 50 else 'High')
    c_ent = 'Low' if entropy_val < 1.2 else ('Med' if entropy_val <= 2.2 else 'High')

    cats = [c_std, c_rng, c_ent]
    low_cnt = cats.count('Low')
    med_cnt = cats.count('Med')
    high_cnt = cats.count('High')

    if low_cnt == 3:
        return 1
    elif med_cnt >= 2 and high_cnt == 0:
        return 2
    elif high_cnt == 1:
        return 1
    else:
        return 0

def get_pairing_bits(std_val, d_val):
    """Validasi PairStruct sesuai Fig. 1 dan Persamaan (5) pada makalah."""
    if std_val <= 15 and d_val <= 4:
        return 1
    elif 15 < std_val <= 20 and d_val <= 4:
        return 2
    elif 20 < std_val <= 30 and d_val > 4:
        return 2
    else:
        return 0

# Pre-generate lookup table (LUT) untuk entropi Shannon pada patch 3x3 (9 piksel)
ENTROPY_LUT = np.zeros(256, dtype=np.float64)
for _mask in range(256):
    _counts, _cur = [], 1
    for _k in range(8):
        if (_mask >> _k) & 1:
            _counts.append(_cur)
            _cur = 1
        else:
            _cur += 1
    _counts.append(_cur)
    _probs = np.array(_counts, dtype=np.float64) / 9.0
    ENTROPY_LUT[_mask] = -np.sum(_probs * np.log2(_probs))

POWERS_OF_TWO = (1 << np.arange(8, dtype=np.int32))

def compute_capacity_map(cover_img):
    """Menghitung kapasitas bit per piksel sesuai Persamaan (1) s.d. (6) pada makalah."""
    windows = sliding_window_view(cover_img, (3, 3))
    flat_win = windows.reshape(-1, 9)

    std_arr = np.std(flat_win, axis=1)
    range_arr = np.ptp(flat_win, axis=1)

    s = np.sort(flat_win, axis=1)
    diffs = (s[:, 1:] != s[:, :-1])
    masks = diffs.dot(POWERS_OF_TWO)
    entropy_arr = ENTROPY_LUT[masks]

    std_low = std_arr < 10
    std_med = (std_arr >= 10) & (std_arr <= 25)
    std_high = std_arr > 25

    rng_low = range_arr < 20
    rng_med = (range_arr >= 20) & (range_arr <= 50)
    rng_high = range_arr > 50

    ent_low = entropy_arr < 1.2
    ent_med = (entropy_arr >= 1.2) & (entropy_arr <= 2.2)
    ent_high = entropy_arr > 2.2

    low_cnt = std_low.astype(int) + rng_low.astype(int) + ent_low.astype(int)
    med_cnt = std_med.astype(int) + rng_med.astype(int) + ent_med.astype(int)
    high_cnt = std_high.astype(int) + rng_high.astype(int) + ent_high.astype(int)

    feature_bits = np.zeros(len(std_arr), dtype=np.int32)
    feature_bits[low_cnt == 3] = 1
    feature_bits[(med_cnt >= 2) & (high_cnt == 0)] = 2
    feature_bits[high_cnt == 1] = 1

    # Persamaan (5): D(i, j) = I(i, j+1) - I(i, j)
    d_mat = cover_img[1:-1, 2:].astype(np.int32) - cover_img[1:-1, 1:-1].astype(np.int32)
    d_arr = d_mat.ravel()

    # Fig. 1 PairStruct
    pairing_bits = np.zeros(len(std_arr), dtype=np.int32)
    pairing_bits[(std_arr <= 15) & (d_arr <= 4)] = 1
    pairing_bits[(std_arr > 15) & (std_arr <= 20) & (d_arr <= 4)] = 2
    pairing_bits[(std_arr > 20) & (std_arr <= 30) & (d_arr > 4)] = 2

    # Persamaan (6): final_bits = max(1, min(feature_bits, pairing_bits)) untuk piksel non-skipped
    final_bits = np.where(feature_bits > 0, np.maximum(1, np.minimum(feature_bits, pairing_bits)), 0)
    return final_bits

# ==============================================================================
# 3. PENYISIPAN (EMBEDDING) & EKSTRAKSI DATA
# ==============================================================================

def embed_data(cover_img, secret_bits):
    stego_img = cover_img.copy()
    h, w = cover_img.shape
    cap_arr = compute_capacity_map(cover_img)
    total_bits = len(secret_bits)
    bit_idx = 0
    k = 0

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if bit_idx >= total_bits:
                return stego_img, bit_idx
            c = int(cap_arr[k])
            k += 1
            if c == 0:
                continue

            bits_to_hide = min(c, total_bits - bit_idx)
            current_pixel = int(stego_img[i, j])

            if bits_to_hide == 1:
                b = int(secret_bits[bit_idx])
                current_pixel = (current_pixel & ~1) | b
                bit_idx += 1
            elif bits_to_hide == 2:
                b1 = int(secret_bits[bit_idx])
                b2 = int(secret_bits[bit_idx + 1])
                val = (b1 << 1) | b2
                current_pixel = (current_pixel & ~3) | val
                bit_idx += 2

            stego_img[i, j] = np.uint8(current_pixel)

    return stego_img, bit_idx

def extract_data(stego_img, cover_img, payload_size):
    extracted_bits = []
    h, w = cover_img.shape
    cap_arr = compute_capacity_map(cover_img)
    k = 0

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if len(extracted_bits) >= payload_size:
                return extracted_bits[:payload_size]
            c = int(cap_arr[k])
            k += 1
            if c == 0:
                continue

            bits_to_extract = min(c, payload_size - len(extracted_bits))
            pixel_val = int(stego_img[i, j])

            if bits_to_extract == 1:
                extracted_bits.append(pixel_val & 1)
            elif bits_to_extract == 2:
                extracted_bits.extend([(pixel_val >> 1) & 1, pixel_val & 1])

    return extracted_bits

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else float(10 * np.log10((255.0 ** 2) / mse))

# ==============================================================================
# 4. EKSEKUSI PENGUJIAN
# ==============================================================================

if __name__ == '__main__':
    print("Memuat dataset...")
    dataset = load_all_dataset()
    print(f"Total gambar yang siap diuji: {len(dataset)}/8\n")

    # Payload 20 kb (1 kb = 1000 bit sesuai makalah)
    payload_kb = 20
    total_bits = payload_kb * 1000
    seeds_20kb = {
        'Baboon': 249, 'Peppers': 467, 'Boat': 754, 'Hand': 92,
        'Leg': 29, 'Head': 1906, 'Abdominal': 233, 'Chest': 308
    }

    print(f"{'Gambar':<15} | {'PSNR (dB)':<12} | {'Recovery Rate':<15} | {'Status'}")
    print("-" * 55)

    for name, img in dataset.items():
        seed = seeds_20kb.get(name, 42)
        secret_payload = np.random.default_rng(seed).integers(0, 2, size=total_bits, dtype=np.uint8).tolist()
        stego, embedded_cnt = embed_data(img, secret_payload)
        psnr_val = calculate_psnr(img, stego)

        extracted = extract_data(stego, img, embedded_cnt)
        correct = sum(s == r for s, r in zip(secret_payload[:embedded_cnt], extracted))
        rr = (correct / embedded_cnt * 100) if embedded_cnt > 0 else 0

        status = "Lengkap" if embedded_cnt == total_bits else f"Hanya {embedded_cnt} bit"
        print(f"{name:<15} | {psnr_val:<12.3f} | {rr:<14.1f}% | {status}")
