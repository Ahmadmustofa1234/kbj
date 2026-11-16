import os
import time
import numpy as np
import cv2
import pydicom
from skimage.metrics import structural_similarity as ssim
from numpy.lib.stride_tricks import sliding_window_view

# ==============================================================================
# 1. PENGATURAN DATASET (USC-SIPI & DICOM)
# ==============================================================================

def load_single_dicom(folder_path):
    """Membaca file DICOM secara rekursif dan mengambil irisan tengah dengan normalisasi yang benar."""
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
        
    target_file = valid_files[len(valid_files) // 2]
    ds = pydicom.dcmread(target_file)
    arr = ds.pixel_array
    
    # Konversi color/RGB jika 3 channel
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float64)
    elif arr.ndim > 2:
        # Jika volume multiframe 3D (misal slices, H, W)
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
    """Memuat 8 citra benchmark: 3 citra alami USC-SIPI dan 5 citra medis DICOM."""
    images = {}
    
    # 3 Citra Alami USC-SIPI
    natural = {
        'Baboon': '4.2.03.tiff',
        'Peppers': '4.2.07.tiff',
        'Boat': 'boat.512.tiff'
    }
    for name, fpath in natural.items():
        if os.path.exists(fpath):
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[Peringatan] OpenCV tidak dapat membaca file gambar: {fpath}")
                continue
            images[name] = cv2.resize(img, (512, 512))
        else:
            print(f"[Peringatan] File {fpath} tidak ditemukan.")
            
    # 5 Citra Medis DICOM
    medical_folders = {
        'Hand': '9ab20df7b89dd84716029b777ed7a5b2',
        'Leg': '02ef8f31ea86a45cfce6eb297c274598',
        'Head': '3b1cda064d9d7f2afffac6dab4b6eaf9',
        'Abdominal': 'a05270ea39afe373939d192e69ef435a',
        'Chest': 'feb6447a72c9a0a31e1bb4459e547964'
    }
    for name, folder in medical_folders.items():
        if os.path.exists(folder):
            dcm_img = load_single_dicom(folder)
            if dcm_img is not None:
                images[name] = cv2.resize(dcm_img, (512, 512))
            else:
                print(f"[Peringatan] DICOM pada folder {folder} gagal dimuat.")
        else:
            print(f"[Peringatan] Folder {folder} tidak ditemukan.")
            
    return images

# ==============================================================================
# 2. VEKTORISASI EKSTRAKSI FITUR & ATURAN HYBRID
# ==============================================================================

# Pre-generate lookup table (LUT) untuk entropi Shannon pada patch 3x3 (9 piksel)
# Ukuran 256 mencakup seluruh pola bitmask transisi nilai terurut
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

def compute_pixel_capacity_map(img, use_paper_formula=True):
    """
    Menghitung peta alokasi bit untuk setiap piksel non-border (1..h-2, 1..w-2).
    Fitur dihitung pada patch 3x3:
      - Standar Deviasi (sigma)
      - Range (R)
      - Entropi (H)
    Ambang batas:
      - sigma: (10, 25)
      - R: (20, 50)
      - H: (1.2, 2.2)
    Hybrid Rules:
      - Case A (semua Low -> 1 bit)
      - Case B (>= 2 Med & 0 High -> 2 bit)
      - Case C (1 High -> 1 bit)
      - Case D (>= 2 High -> 0 bit / dilewati)
    PairStruct:
      - D = |I(i, j+1) - I(i, j)|
      - sigma <= 15 and D <= 4 -> 1 bit
      - 15 < sigma <= 30 and D <= 4 -> 2 bit
      - Else -> 0 bit
    Kombinasi Akhir:
      - Sesuai Persamaan (6) Paper: max(1, min(feature_bits, pairing_bits)) untuk piksel tidak dilewati (Case D = 0)
    """
    windows = sliding_window_view(img, (3, 3))
    flat_win = windows.reshape(-1, 9)
    
    # 1. Standar Deviasi
    std_arr = np.std(flat_win, axis=1)
    
    # 2. Range
    range_arr = np.ptp(flat_win, axis=1)
    
    # 3. Entropi Shannon cepat via LUT
    s = np.sort(flat_win, axis=1)
    diffs = (s[:, 1:] != s[:, :-1])
    masks = diffs.dot(POWERS_OF_TWO)
    entropy_arr = ENTROPY_LUT[masks]
    
    # Klasifikasi Fitur
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
    feature_bits[low_cnt == 3] = 1                         # Case A
    feature_bits[(med_cnt >= 2) & (high_cnt == 0)] = 2     # Case B
    feature_bits[high_cnt == 1] = 1                         # Case C
    # Case D (high_cnt >= 2) tetap 0 (dilewati)
    
    # PairStruct
    d_mat = np.abs(img[1:-1, 2:].astype(np.int32) - img[1:-1, 1:-1].astype(np.int32))
    d_arr = d_mat.ravel()
    
    pairing_bits = np.zeros(len(std_arr), dtype=np.int32)
    pairing_bits[(std_arr <= 15) & (d_arr <= 4)] = 1
    pairing_bits[(std_arr > 15) & (std_arr <= 30) & (d_arr <= 4)] = 2
    
    if use_paper_formula:
        # Persamaan (6) di paper: untuk piksel non-skipped (feature_bits > 0), max(1, min(fb, pb))
        final_bits = np.where(feature_bits > 0, np.maximum(1, np.minimum(feature_bits, pairing_bits)), 0)
    else:
        final_bits = np.minimum(feature_bits, pairing_bits)
        
    return final_bits

# ==============================================================================
# 3. PENYISIPAN (EMBEDDING) & EKSTRAKSI DATA
# ==============================================================================

def embed_data(cover_img, secret_bits, cap_arr):
    """Menyisipkan bit rahasia ke dalam LSB piksel sesuai alokasi bit kapasitas."""
    stego_img = cover_img.copy()
    h, w = cover_img.shape
    total_bits = len(secret_bits)
    bit_idx = 0
    k = 0
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if bit_idx >= total_bits:
                return stego_img, bit_idx
            c = cap_arr[k]
            k += 1
            if c == 0:
                continue
                
            bits_to_hide = min(c, total_bits - bit_idx)
            current_pixel = int(stego_img[i, j])
            
            if bits_to_hide == 1:
                b = secret_bits[bit_idx]
                current_pixel = (current_pixel & ~1) | b
                bit_idx += 1
            elif bits_to_hide == 2:
                b1 = secret_bits[bit_idx]
                b2 = secret_bits[bit_idx + 1]
                val = (b1 << 1) | b2
                current_pixel = (current_pixel & ~3) | val
                bit_idx += 2
                
            stego_img[i, j] = np.uint8(current_pixel)
            
    return stego_img, bit_idx

def extract_data(stego_img, cap_arr, payload_size):
    """Mengekstraksi bit rahasia dari LSB citra stego."""
    extracted_bits = []
    h, w = stego_img.shape
    k = 0
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if len(extracted_bits) >= payload_size:
                return extracted_bits[:payload_size]
            c = cap_arr[k]
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

def calculate_ssim(img1, img2):
    return float(ssim(img1, img2, data_range=255))

# ==============================================================================
# 4. BENCHMARK REFERENSI MAKALAH
# ==============================================================================

PAPER_TABLE_I = {
    'Baboon':    [75.432, 65.334, 62.379, 60.552, 60.552, 58.362, 57.558, 56.872, 56.266, 55.780, 55.320],
    'Peppers':   [75.165, 65.347, 62.337, 60.559, 59.292, 58.348, 57.567, 56.875, 56.313, 55.813, 55.330],
    'Boat':      [75.258, 65.305, 62.351, 60.567, 59.304, 58.367, 57.525, 56.851, 56.287, 55.803, 55.345],
    'Hand':      [75.405, 65.302, 62.293, 60.575, 59.317, 58.354, 57.556, 56.844, 56.284, 55.772, 55.322],
    'Leg':       [75.327, 65.314, 62.286, 60.549, 59.302, 58.333, 57.536, 56.898, 56.297, 55.782, 55.334],
    'Head':      [75.353, 65.032, 62.198, 60.476, 59.195, 58.272, 57.468, 56.812, 56.263, 55.755, 55.298],
    'Abdominal': [75.477, 65.198, 62.292, 60.544, 59.311, 58.334, 57.531, 56.857, 56.316, 55.793, 55.320],
    'Chest':     [75.441, 65.248, 62.276, 60.540, 59.314, 58.370, 57.514, 56.873, 56.294, 55.776, 55.312],
}

PAPER_TABLE_II = {
    # kb: (Prior PSNR, Prior SSIM, Prior RR)
    40:  (59.162, 1.000, 100),
    50:  (57.626, 1.000, 100),
    60:  (56.856, 1.000, 100),
    70:  (56.158, 1.000, 100),
    80:  (55.629, 1.000, 100),
    90:  (55.208, 1.000, 100),
    100: (54.832, 1.000, 100),
}

PAPER_TABLE_III = {
    # kb: (Prior PSNR, Prior SSIM, Prior RR)
    40:  (58.626, None, None),
    50:  (57.656, None, None),
    60:  (56.854, None, None),
    70:  (56.251, None, None),
    80:  (55.749, None, None),
    90:  (55.205, None, None),
    100: (54.715, None, None),
}

# ==============================================================================
# 5. EKSEKUSI PENGUJIAN MENYELURUH & FORMAT TABEL MAKALAH
# ==============================================================================

def run_comprehensive_evaluation():
    print("=" * 115)
    print("PENGUJIAN STEGANOGRAFI MULTI-PARAMETER TEKSTUR BERDASARKAN MAKALAH")
    print("Sarah Alissa Putri & Tohari Ahmad (ICICyTA)")
    print("=" * 115)
    
    t_start = time.time()
    print("\n[1/3] Memuat seluruh 8 citra dataset (USC-SIPI & DICOM)...")
    dataset = load_all_dataset()
    print(f"-> Berhasil memuat {len(dataset)}/8 citra: {list(dataset.keys())}\n")
    
    order = ['Baboon', 'Peppers', 'Boat', 'Hand', 'Leg', 'Head', 'Abdominal', 'Chest']
    payloads_kb = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # Pre-generate payload biner acak dengan seed tetap agar deterministik
    np.random.seed(42)
    max_payload = np.random.randint(0, 2, max(payloads_kb) * 1024).tolist()
    payload_cache = {
        kb: max_payload[:kb * 1024]
        for kb in payloads_kb
    }
        
    # Pra-hitung peta alokasi kapasitas per citra
    print("[2/3] Menghitung peta fitur tekstur & alokasi bit per citra...")
    cap_maps = {}
    for name in order:
        if name in dataset:
            cap_maps[name] = compute_pixel_capacity_map(dataset[name], use_paper_formula=True)
            total_cap_kb = np.sum(cap_maps[name]) / 1024.0
            print(f"   * {name:<10}: Kapasitas maksimum = {total_cap_kb:.1f} kb")
            
    print("\n[3/3] Menjalankan penyisipan, ekstraksi, dan perhitungan metrik...")
    
    # Simpan hasil untuk seluruh citra dan seluruh payload
    # metrics_results[img_name][kb] = (psnr, ssim, rr)
    metrics_results = {name: {} for name in order}
    
    for name in order:
        if name not in dataset:
            continue
        cover = dataset[name]
        cap = cap_maps[name]
        
        for kb in payloads_kb:
            secret = payload_cache[kb]
            stego, n_emb = embed_data(cover, secret, cap)
            psnr_val = calculate_psnr(cover, stego)
            
            # Ekstraksi dan evaluasi Recovery Rate
            ext_bits = extract_data(stego, cap, n_emb)
            n_match = sum(s == r for s, r in zip(secret[:n_emb], ext_bits))
            rr_val = (n_match / n_emb * 100.0) if n_emb > 0 else 0.0
            
            # SSIM dihitung untuk perbandingan
            ssim_val = calculate_ssim(cover, stego)
            
            metrics_results[name][kb] = {
                'psnr': psnr_val,
                'ssim': ssim_val,
                'rr': rr_val,
                'embedded': n_emb
            }
            
    # ==========================================================================
    # CETAK TABEL 1 (TABLE I DI PAPER)
    # ==========================================================================
    print("\n" + "=" * 115)
    print("TABEL 1. NILAI PSNR TERHADAP PAYLOAD DAN CITRA COVER PADA METODE YANG DIUSULKAN")
    print("(Format persis seperti TABLE I pada Makalah)")
    print("=" * 115)
    
    header = f"{'Cover Image':<12} | " + " | ".join(f"{kb:>6} kb" for kb in payloads_kb)
    print(header)
    print("-" * len(header))
    
    for name in order:
        if name in metrics_results and metrics_results[name]:
            row_vals = []
            for kb in payloads_kb:
                val = metrics_results[name][kb]['psnr']
                row_vals.append(f"{val:>9.3f}")
            print(f"{name:<12} | " + " | ".join(row_vals))
            
    print("-" * len(header))
    print("Catatan: Rentang payload 1 kb sampai 100 kb (1 kb = 1024 bit).")
    
    # ==========================================================================
    # CETAK TABEL 2 (TABLE II DI PAPER) - BABOON
    # ==========================================================================
    print("\n" + "=" * 115)
    print("TABEL 2. PERBANDINGAN NILAI PSNR, SSIM, DAN RR KHUSUS CITRA BABOON")
    print("(Format persis seperti TABLE II pada Makalah: Perbandingan Prior [7] vs Proposed)")
    print("=" * 115)
    
    t2_head = (
        f"{'Payload':<8} | "
        f"{'PSNR Prior [7]':<15} | {'PSNR Proposed':<15} | "
        f"{'SSIM Prior [7]':<15} | {'SSIM Proposed':<15} | "
        f"{'RR Prior (%)':<13} | {'RR Proposed (%)':<15}"
    )
    print(t2_head)
    print("-" * len(t2_head))
    
    for kb in [40, 50, 60, 70, 80, 90, 100]:
        prior_p, prior_s, prior_r = PAPER_TABLE_II[kb]
        res = metrics_results['Baboon'][kb]
        
        prior_p_str = f"{prior_p:.3f} dB"
        prop_p_str = f"{res['psnr']:.3f} dB"
        prior_s_str = f"{prior_s:.3f}"
        prop_s_str = f"{res['ssim']:.3f}"
        prior_r_str = f"{prior_r}%"
        prop_r_str = f"{res['rr']:.1f}%"
        
        print(
            f"{kb:>3} kb    | "
            f"{prior_p_str:<15} | {prop_p_str:<15} | "
            f"{prior_s_str:<15} | {prop_s_str:<15} | "
            f"{prior_r_str:<13} | {prop_r_str:<15}"
        )
        
    print("-" * len(t2_head))
    
    # ==========================================================================
    # CETAK TABEL 3 (TABLE III DI PAPER) - HEAD
    # ==========================================================================
    print("\n" + "=" * 115)
    print("TABEL 3. PERBANDINGAN NILAI PSNR, SSIM, DAN RR KHUSUS CITRA HEAD")
    print("(Format persis seperti TABLE III pada Makalah: Perbandingan Prior [7] vs Proposed)")
    print("=" * 115)
    
    t3_head = (
        f"{'Payload':<8} | "
        f"{'PSNR Prior [7]':<15} | {'PSNR Proposed':<15} | "
        f"{'SSIM Prior [7]':<15} | {'SSIM Proposed':<15} | "
        f"{'RR Prior (%)':<13} | {'RR Proposed (%)':<15}"
    )
    print(t3_head)
    print("-" * len(t3_head))
    
    for kb in [40, 50, 60, 70, 80, 90, 100]:
        prior_p, prior_s, prior_r = PAPER_TABLE_III[kb]
        res = metrics_results['Head'][kb]
        
        prior_p_str = f"{prior_p:.3f} dB"
        prop_p_str = f"{res['psnr']:.3f} dB"
        prior_s_str = f"{prior_s:.3f}" if prior_s is not None else "-"
        prop_s_str = f"{res['ssim']:.3f}"
        prior_r_str = f"{prior_r}%" if prior_r is not None else "-"
        prop_r_str = f"{res['rr']:.1f}%"
        
        print(
            f"{kb:>3} kb    | "
            f"{prior_p_str:<15} | {prop_p_str:<15} | "
            f"{prior_s_str:<15} | {prop_s_str:<15} | "
            f"{prior_r_str:<13} | {prop_r_str:<15}"
        )
        
    print("-" * len(t3_head))
    
    t_end = time.time()
    print(f"\nPengujian selesai dalam {t_end - t_start:.2f} detik.")
    print("=" * 115)

if __name__ == '__main__':
    run_comprehensive_evaluation()
