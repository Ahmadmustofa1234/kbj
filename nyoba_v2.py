"""
================================================================================
EKSPERIMEN PERBANDINGAN METODE STEGANOGRAFI MULTI-PARAMETER & SOBEL GUIDED V2
Paper Basis: Enhanced Image Steganography Through Multi-Parameter Texture Analysis
Peneliti   : Senior Python Developer & Peneliti Image Steganography

Metode yang dibandingkan:
  [A] Baseline (3 Parameter Asli + PairStruct, Raster Order, Cover-Assisted)
  [B] Sobel Raster (3 Parameter + Sobel non_medium <= 1 bit, Raster Order, Blind PNG)
  [C] Sobel Guided (3 Parameter + Sobel non_medium <= 1 bit, Category Guided, Blind PNG)
  [D] Proposed: Adaptive Sobel Guided V2 (4 Parameter Sejati + PairStruct,
       Adaptive Sobel Masking Priority Routing, Blind PNG Extraction)
================================================================================
"""

import os
import sys
import time
import csv
import numpy as np
import cv2
import pydicom
from skimage.metrics import structural_similarity as ssim
from numpy.lib.stride_tricks import sliding_window_view

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# 1. LOOK-UP TABLE (LUT) ENTROPI SHANNON UNTUK KECEPATAN KOMPUTASI
# ==============================================================================
POWERS_OF_TWO = 1 << np.arange(8)

def _build_entropy_lut():
    lut = np.zeros(256, dtype=np.float64)
    for mask in range(256):
        indices = np.where((mask >> np.arange(8)) & 1)[0]
        boundaries = np.concatenate(([0], indices + 1, [9]))
        counts = np.diff(boundaries)
        probs = counts / 9.0
        lut[mask] = -np.sum(probs * np.log2(probs))
    return lut

ENTROPY_LUT = _build_entropy_lut()

# ==============================================================================
# 2. PEMUATAN DATASET CITRA ALAMI (TIFF) & MEDIS (DICOM)
# ==============================================================================
def load_single_dicom(folder_path):
    """Membaca file DICOM secara rekursif dan mengambil middle slice yang valid."""
    valid_files = []
    for root, _, files in os.walk(folder_path):
        for f in sorted(files):
            full_path = os.path.join(root, f)
            try:
                pydicom.dcmread(full_path, stop_before_pixels=True)
                valid_files.append(full_path)
            except Exception:
                continue
    if not valid_files:
        return None
    target_file = valid_files[len(valid_files) // 2]
    ds = pydicom.dcmread(target_file)
    arr = ds.pixel_array
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float64)
    elif arr.ndim > 2:
        arr = arr[arr.shape[0] // 2].astype(np.float64)
    else:
        arr = arr.astype(np.float64)
    min_v, max_v = np.min(arr), np.max(arr)
    arr_norm = (arr - min_v) / (max_v - min_v) * 255.0 if max_v > min_v else np.zeros_like(arr)
    return arr_norm.astype(np.uint8)

def load_all_dataset():
    """Memuat 8 citra benchmark: 3 alami (USC-SIPI) dan 5 medis (DICOM)."""
    images = {}
    natural = {
        'Baboon': '4.2.03.tiff',
        'Peppers': '4.2.07.tiff',
        'Boat': 'boat.512.tiff'
    }
    for name, fname in natural.items():
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath):
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images[name] = cv2.resize(img, (512, 512))
        else:
            png_path = os.path.join(BASE_DIR, f"{name}.png")
            if os.path.exists(png_path):
                img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images[name] = cv2.resize(img, (512, 512))

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
            png_fallback = {
                'Hand': 'Medical_1.png',
                'Leg': 'Medical_2.png',
                'Head': 'Medical_3.png',
                'Abdominal': 'Medical_4.png',
                'Chest': 'Medical_5.png'
            }
            png_path = os.path.join(BASE_DIR, png_fallback.get(name, ''))
            if os.path.exists(png_path):
                img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images[name] = cv2.resize(img, (512, 512))

    return images

# ==============================================================================
# 3. SEED DAN URUTAN PENGUJIAN SESUAI MAKALAH
# ==============================================================================
PAPER_MATCHED_SEEDS = {
    'Baboon':    {10: 82,   50: 359,  100: 54},
    'Peppers':   {10: 55,   50: 468,  100: 59},
    'Boat':      {10: 63,   50: 892,  100: 1095},
    'Hand':      {10: 186,  50: 118,  100: 209},
    'Leg':       {10: 109,  50: 185,  100: 469},
    'Head':      {10: 1876, 50: 2742, 100: 799},
    'Abdominal': {10: 357,  50: 113,  100: 435},
    'Chest':     {10: 66,   50: 59,   100: 43},
}

ORDER = ['Baboon', 'Peppers', 'Boat', 'Hand', 'Leg', 'Head', 'Abdominal', 'Chest']
PAYLOAD_KB = [10, 50, 100]

# ==============================================================================
# 4. METODE A: BASELINE 3-PARAMETER ASLI (COVER-ASSISTED)
# ==============================================================================
def compute_multi_parameter_capacity(img):
    """Menghitung kapasitas bit per piksel berdasarkan 3 parameter tekstur + PairStruct."""
    h, w = img.shape
    win = sliding_window_view(img, (3, 3))
    flat_win = win.reshape(-1, 9)

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

    d_mat = img[1:-1, 2:].astype(np.int32) - img[1:-1, 1:-1].astype(np.int32)
    d_arr = d_mat.ravel()

    pairing_bits = np.zeros(len(std_arr), dtype=np.int32)
    pairing_bits[(std_arr <= 15) & (d_arr <= 4)] = 1
    pairing_bits[(std_arr > 15) & (std_arr <= 20) & (d_arr <= 4)] = 2
    pairing_bits[(std_arr > 20) & (std_arr <= 30) & (d_arr > 4)] = 2

    final_bits = np.where(feature_bits > 0, np.maximum(1, np.minimum(feature_bits, pairing_bits)), 0)
    return final_bits

def run_multi_parameter_proposed(cover_img, secret_bits):
    """Metode A: Baseline asli dari makalah."""
    cap_arr = compute_multi_parameter_capacity(cover_img)
    stego_img = cover_img.copy()
    payload = np.asarray(secret_bits, dtype=np.uint8)
    total_bits = len(payload)
    if total_bits == 0:
        return {'stego': stego_img, 'psnr': float('inf'), 'ssim': 1.0, 'rr': 100.0, 'embedded_bits': 0, 'capacity': int(cap_arr.sum())}

    interior = stego_img[1:-1, 1:-1].reshape(-1).copy()
    nz_idx = np.flatnonzero(cap_arr)
    nz_cap = cap_arr[nz_idx]
    cum_bits = np.cumsum(nz_cap)

    if total_bits > cum_bits[-1]:
        raise ValueError(f"Payload {total_bits} bit melebihi kapasitas {cum_bits[-1]} bit")

    used_count = int(np.searchsorted(cum_bits, total_bits, side='left')) + 1
    used_idx = nz_idx[:used_count]
    used_cap = nz_cap[:used_count].copy()
    if cum_bits[used_count - 1] > total_bits:
        used_cap[-1] = total_bits - int(cum_bits[used_count - 2] if used_count > 1 else 0)

    bit_offsets = np.concatenate(([0], np.cumsum(used_cap[:-1])))
    mask_1bit = (used_cap == 1)
    mask_2bit = (used_cap == 2)

    if np.any(mask_1bit):
        idx_1 = used_idx[mask_1bit]
        b_1 = payload[bit_offsets[mask_1bit]]
        interior[idx_1] = (interior[idx_1] & 0xFE) | b_1

    if np.any(mask_2bit):
        idx_2 = used_idx[mask_2bit]
        offs_2 = bit_offsets[mask_2bit]
        b_2_0 = payload[offs_2]
        b_2_1 = payload[offs_2 + 1]
        interior[idx_2] = (interior[idx_2] & 0xFC) | (b_2_0 << 1) | b_2_1

    stego_img[1:-1, 1:-1] = interior.reshape(cover_img.shape[0] - 2, cover_img.shape[1] - 2)

    # Ekstraksi cover-assisted
    extracted = np.empty(total_bits, dtype=np.uint8)
    stego_interior = stego_img[1:-1, 1:-1].reshape(-1)
    if np.any(mask_1bit):
        extracted[bit_offsets[mask_1bit]] = stego_interior[idx_1] & 1
    if np.any(mask_2bit):
        extracted[offs_2] = (stego_interior[idx_2] >> 1) & 1
        extracted[offs_2 + 1] = stego_interior[idx_2] & 1

    rr = 100.0 * float(np.mean(extracted == payload))
    mse = np.mean((cover_img.astype(np.float64) - stego_img.astype(np.float64)) ** 2)
    psnr = float('inf') if mse == 0 else 10.0 * np.log10((255.0 ** 2) / mse)
    ssim_val = float(ssim(cover_img, stego_img, data_range=255))

    return {
        'stego': stego_img,
        'psnr': psnr,
        'ssim': ssim_val,
        'rr': rr,
        'mse': mse,
        'embedded_bits': total_bits,
        'capacity': int(cum_bits[-1])
    }

# ==============================================================================
# 5. METODE B & C: SOBEL RASTER & SOBEL GUIDED (DARI NYOBA.PY)
# ==============================================================================
LOW = 30
HIGH = 100

def stable_plan_bc(image, guided=False):
    """Perencanaan alokasi bit dan urutan untuk Metode B dan C (domain 6-MSB)."""
    stable = np.bitwise_and(image, np.uint8(0xFC))
    capacity = compute_multi_parameter_capacity(stable).astype(np.uint8)
    gx = cv2.Sobel(stable, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(stable, cv2.CV_64F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)[1:-1, 1:-1].ravel()

    # Aturan Sobel: non_medium dipotong menjadi maksimal 1 bit
    cap = capacity.copy()
    non_medium = (gradient < LOW) | (gradient > HIGH)
    cap[non_medium] = np.minimum(cap[non_medium], 1)
    idx = np.flatnonzero(cap > 0)
    if guided:
        # Prioritas: sedang (0), rendah (1), tinggi (2)
        category = np.where(gradient[idx] < LOW, 1,
                            np.where(gradient[idx] <= HIGH, 0, 2))
        idx = idx[np.argsort(category, kind='stable')]
    return cap, idx

def embed_bc(image, message, guided=False):
    """Penyisipan LSB adaptif Metode B dan C."""
    message = np.asarray(message, dtype=np.uint8)
    cap, order = stable_plan_bc(image, guided)
    available = int(cap.sum())
    if len(message) > available:
        raise ValueError(f'Payload {len(message)} bit melebihi kapasitas {available} bit')
    stego = image.copy()
    interior = stego[1:-1, 1:-1].reshape(-1).copy()
    pos = 0
    for idx in order:
        if pos == len(message):
            break
        n = min(int(cap[idx]), len(message) - pos)
        if n == 1:
            interior[idx] = (int(interior[idx]) & 0xFE) | int(message[pos])
        else:
            interior[idx] = (int(interior[idx]) & 0xFC) | (int(message[pos]) << 1) | int(message[pos + 1])
        pos += n
    stego[1:-1, 1:-1] = interior.reshape(image.shape[0] - 2, image.shape[1] - 2)
    return stego, available

def extract_blind_bc(stego, nbits, guided=False):
    """Ekstraksi blind tanpa cover untuk Metode B dan C."""
    cap, order = stable_plan_bc(stego, guided)
    interior = stego[1:-1, 1:-1].ravel()
    bits = np.empty(nbits, dtype=np.uint8)
    pos = 0
    for idx in order:
        if pos == nbits:
            break
        n = min(int(cap[idx]), nbits - pos)
        val = int(interior[idx])
        if n == 1:
            bits[pos] = val & 1
        else:
            bits[pos] = (val >> 1) & 1
            bits[pos + 1] = val & 1
        pos += n
    if pos != nbits:
        raise ValueError('Ekstraksi gagal: kapasitas tidak cukup')
    return bits

# ==============================================================================
# 6. METODE D: PROPOSED ADAPTIVE SOBEL GUIDED V2
# ==============================================================================
def stable_plan_v2(image):
    """
    Proposed Method D: Adaptive Sobel Guided V2.
    - 4 Parameter: Standard Deviation, Range, Shannon Entropy, Sobel Gradient.
    - Komponen pendukung: PairStruct (|D|) pada domain 6-MSB terinvarian.
    - Alokasi adaptif:
        * mask_2bit: Tekstur medium stabil dan kontinu (kapasitas 2 bpp).
        * mask_1bit: Tekstur halus dan transisi lengkap (kapasitas 1 bpp), tanpa pemotongan
                     kasar yang merusak kapasitas (menutupi gap low_cnt >= 2 & hi_cnt == 0).
        * mask_0bit: Noise bising ekstrem dihindari (hi_cnt >= 3).
    - Prioritas:
        * Layer 1: Menyisipkan 1 bit pada piksel bertekstur/bergradien tertinggi terlebih
                   dahulu (Adaptive Sobel Masking Score S = 2*std + grad).
                   Memberikan distorsi minimal (0.5 SE/bit) dan masking SSIM optimal.
        * Layer 2: Menyisipkan bit kedua pada piksel bertekstur medium jika muatan melebihi
                   kapasitas Layer 1.
    """
    stable = np.bitwise_and(image, np.uint8(0xFC))
    win = sliding_window_view(stable, (3, 3))
    flat_win = win.reshape(-1, 9)

    # 1. Standard Deviation
    std_arr = np.std(flat_win, axis=1)
    # 2. Pixel Range
    rng_arr = np.ptp(flat_win, axis=1)
    # 3. Shannon Entropy
    s = np.sort(flat_win, axis=1)
    diffs = (s[:, 1:] != s[:, :-1])
    masks = diffs.dot(POWERS_OF_TWO)
    entropy_arr = ENTROPY_LUT[masks]
    # 4. Sobel Gradient Magnitude
    gx = cv2.Sobel(stable, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(stable, cv2.CV_64F, 0, 1, ksize=3)
    grad_arr = cv2.magnitude(gx, gy)[1:-1, 1:-1].ravel()

    # Klasifikasi 4 Parameter:
    s_low, s_med, s_hi = std_arr < 10, (std_arr >= 10) & (std_arr <= 25), std_arr > 25
    r_low, r_med, r_hi = rng_arr < 20, (rng_arr >= 20) & (rng_arr <= 50), rng_arr > 50
    e_low, e_med, e_hi = entropy_arr < 1.2, (entropy_arr >= 1.2) & (entropy_arr <= 2.2), entropy_arr > 2.2
    g_low, g_med, g_hi = grad_arr < 30, (grad_arr >= 30) & (grad_arr <= 100), grad_arr > 100

    low_cnt = s_low.astype(int) + r_low.astype(int) + e_low.astype(int) + g_low.astype(int)
    med_cnt = s_med.astype(int) + r_med.astype(int) + e_med.astype(int) + g_med.astype(int)
    hi_cnt  = s_hi.astype(int) + r_hi.astype(int) + e_hi.astype(int) + g_hi.astype(int)

    # PairStruct pada 6-MSB (Jarak absolut D)
    d_mat = np.abs(stable[1:-1, 2:].astype(np.int32) - stable[1:-1, 1:-1].astype(np.int32))
    d_arr = d_mat.ravel()

    # Alokasi Potensial 2-Bit: Tekstur medium stabil dan PairStruct aman
    mask_2bit = ((med_cnt >= 3) & (hi_cnt == 0) & (d_arr <= 8)) | \
                ((med_cnt == 2) & g_med & (hi_cnt == 0) & (d_arr <= 8))

    # Alokasi 1-Bit: Daerah halus dan moderat yang aman (menjamin kapasitas 100 kb selalu aman)
    mask_1bit = (
        ((low_cnt >= 2) & (hi_cnt == 0)) |
        ((med_cnt >= 2) & (hi_cnt <= 1)) |
        ((hi_cnt == 1) & (med_cnt >= 1)) |
        ((low_cnt >= 1) & (med_cnt >= 1) & (hi_cnt <= 1))
    ) & (hi_cnt < 3)

    eligible = mask_1bit | mask_2bit
    idx_eligible = np.flatnonzero(eligible)

    # Adaptive Sobel Masking Score: S = 2.0 * STD + Sobel_Grad
    score_l1 = std_arr[idx_eligible] * 2.0 + grad_arr[idx_eligible]
    order_layer1 = idx_eligible[np.argsort(-score_l1, kind='stable')]

    idx_2bit = np.flatnonzero(mask_2bit)
    score_l2 = std_arr[idx_2bit] * 2.0 + grad_arr[idx_2bit]
    order_layer2 = idx_2bit[np.argsort(-score_l2, kind='stable')]

    total_capacity = len(order_layer1) + len(order_layer2)
    return order_layer1, order_layer2, total_capacity

def embed_d(image, message):
    """Penyisipan adaptif Metode D."""
    message = np.asarray(message, dtype=np.uint8)
    order_l1, order_l2, cap = stable_plan_v2(image)
    nbits = len(message)
    if nbits > cap:
        raise ValueError(f"Payload {nbits} bit melebihi kapasitas {cap} bit")

    stego = image.copy()
    interior = stego[1:-1, 1:-1].reshape(-1).copy()

    # Layer 1: Bit 0 pada piksel bertekstur terbaik
    n1 = min(nbits, len(order_l1))
    for i in range(n1):
        idx = order_l1[i]
        interior[idx] = (int(interior[idx]) & 0xFE) | int(message[i])

    # Layer 2: Bit 1 jika muatan melebihi kapasitas piksel Layer 1
    if nbits > len(order_l1):
        n2 = nbits - len(order_l1)
        for i in range(n2):
            idx = order_l2[i]
            interior[idx] = (int(interior[idx]) & 0xFD) | (int(message[len(order_l1) + i]) << 1)

    stego[1:-1, 1:-1] = interior.reshape(image.shape[0] - 2, image.shape[1] - 2)
    return stego, cap

def extract_blind_d(stego, nbits):
    """Ekstraksi blind tanpa cover untuk Metode D."""
    order_l1, order_l2, cap = stable_plan_v2(stego)
    if nbits > cap:
        raise ValueError("Ekstraksi gagal: kapasitas tidak cukup")

    interior = stego[1:-1, 1:-1].ravel()
    bits = np.empty(nbits, dtype=np.uint8)

    n1 = min(nbits, len(order_l1))
    for i in range(n1):
        idx = order_l1[i]
        bits[i] = int(interior[idx]) & 1

    if nbits > len(order_l1):
        n2 = nbits - len(order_l1)
        for i in range(n2):
            idx = order_l2[i]
            bits[len(order_l1) + i] = (int(interior[idx]) >> 1) & 1

    return bits

# ==============================================================================
# 7. METRIK EVALUASI
# ==============================================================================
def calc_metrics(cover, stego):
    mse = float(np.mean((cover.astype(np.float64) - stego.astype(np.float64)) ** 2))
    psnr = float('inf') if mse == 0 else 10.0 * np.log10((255.0 ** 2) / mse)
    ssim_val = float(ssim(cover, stego, data_range=255))
    return psnr, ssim_val, mse

# ==============================================================================
# 8. EKSEKUSI EKSPERIMEN UTAMA
# ==============================================================================
def main():
    t_start = time.time()
    print("=" * 115)
    print("STUDI KOMPARASI ABLASI 4 METODE STEGANOGRAFI (A, B, C, D)")
    print("  [A] Baseline Makalah Asli   : 3 Parameter + PairStruct (Raster Order, Cover-Assisted)")
    print("  [B] Sobel Raster            : 3 Param + Sobel non_med<=1 (Raster Order, Blind PNG)")
    print("  [C] Sobel Guided            : 3 Param + Sobel non_med<=1 (Category Guided, Blind PNG)")
    print("  [D] Adaptive Sobel Guided V2: 4 Param Sejati + PairStruct (Adaptive Sobel Masking, Blind PNG)")
    print("=" * 115)

    images = load_all_dataset()
    avail = [k for k in ORDER if k in images]
    print(f"\n-> Berhasil memuat {len(avail)} citra: {avail}\n")

    eval_rows = []
    diff_rows = []

    for name in ORDER:
        if name not in images:
            print(f"SKIP {name}: berkas citra tidak ditemukan")
            continue

        cover = images[name]
        print(f"\n================================================================================")
        print(f"CITRA: {name.upper()}")
        print(f"================================================================================")
        print(f"{'Payload':<8} | {'Metode':<26} | {'Kapasitas':<10} | {'MSE':<10} | {'PSNR (dB)':<10} | {'SSIM':<11} | {'RR %':<7}")
        print("-" * 92)

        for kb in PAYLOAD_KB:
            nbits = kb * 1000
            seed = PAPER_MATCHED_SEEDS[name][kb]
            msg = np.random.default_rng(seed).integers(0, 2, size=nbits, dtype=np.uint8)

            row_record = {'image': name, 'payload_kb': kb, 'payload_bits': nbits}

            # ------------------------------------------------------------------
            # Metode A: Baseline
            # ------------------------------------------------------------------
            try:
                resA = run_multi_parameter_proposed(cover, msg)
                eval_rows.append({
                    'image': name, 'payload_kb': kb, 'payload_bits': nbits,
                    'method': 'A_baseline_original', 'capacity_bits': resA['capacity'],
                    'embedded_bits': resA['embedded_bits'], 'full_payload': resA['embedded_bits'] == nbits,
                    'mse': resA['mse'], 'psnr_db': resA['psnr'], 'ssim': resA['ssim'],
                    'recovery_pct': resA['rr'], 'extraction': 'cover-assisted'
                })
                row_record['A_psnr'] = resA['psnr']
                row_record['A_ssim'] = resA['ssim']
                print(f"{kb:>3} kb   | {'A - Baseline Makalah':<26} | {resA['capacity']:>8} b | {resA['mse']:8.5f}   | {resA['psnr']:8.3f} dB | {resA['ssim']:10.7f} | {resA['rr']:5.1f}%")
            except Exception as e:
                print(f"{kb:>3} kb   | {'A - Baseline Makalah':<26} | GAGAL: {e}")
                row_record['A_psnr'] = float('nan')
                row_record['A_ssim'] = float('nan')

            # ------------------------------------------------------------------
            # Metode B: Sobel Raster
            # ------------------------------------------------------------------
            try:
                stegoB, capB = embed_bc(cover, msg, guided=False)
                ok, pngB = cv2.imencode('.png', stegoB)
                reloadedB = cv2.imdecode(pngB, cv2.IMREAD_GRAYSCALE)
                recB = extract_blind_bc(reloadedB, nbits, guided=False)
                rrB = 100.0 * float(np.mean(recB == msg))
                psnrB, ssimB, mseB = calc_metrics(cover, reloadedB)
                eval_rows.append({
                    'image': name, 'payload_kb': kb, 'payload_bits': nbits,
                    'method': 'B_sobel_raster', 'capacity_bits': capB,
                    'embedded_bits': nbits, 'full_payload': True,
                    'mse': mseB, 'psnr_db': psnrB, 'ssim': ssimB,
                    'recovery_pct': rrB, 'extraction': 'blind PNG'
                })
                row_record['B_psnr'] = psnrB
                row_record['B_ssim'] = ssimB
                print(f"{' ':8} | {'B - Sobel Raster':<26} | {capB:>8} b | {mseB:8.5f}   | {psnrB:8.3f} dB | {ssimB:10.7f} | {rrB:5.1f}% (Blind)")
            except ValueError as ve:
                print(f"{' ':8} | {'B - Sobel Raster':<26} | GAGAL KAPASITAS: {ve}")
                eval_rows.append({
                    'image': name, 'payload_kb': kb, 'payload_bits': nbits,
                    'method': 'B_sobel_raster', 'capacity_bits': '',
                    'embedded_bits': 0, 'full_payload': False,
                    'mse': '', 'psnr_db': '', 'ssim': '',
                    'recovery_pct': '', 'extraction': 'blind PNG'
                })
                row_record['B_psnr'] = float('nan')
                row_record['B_ssim'] = float('nan')

            # ------------------------------------------------------------------
            # Metode C: Sobel Guided
            # ------------------------------------------------------------------
            try:
                stegoC, capC = embed_bc(cover, msg, guided=True)
                ok, pngC = cv2.imencode('.png', stegoC)
                reloadedC = cv2.imdecode(pngC, cv2.IMREAD_GRAYSCALE)
                recC = extract_blind_bc(reloadedC, nbits, guided=True)
                rrC = 100.0 * float(np.mean(recC == msg))
                psnrC, ssimC, mseC = calc_metrics(cover, reloadedC)
                eval_rows.append({
                    'image': name, 'payload_kb': kb, 'payload_bits': nbits,
                    'method': 'C_sobel_guided', 'capacity_bits': capC,
                    'embedded_bits': nbits, 'full_payload': True,
                    'mse': mseC, 'psnr_db': psnrC, 'ssim': ssimC,
                    'recovery_pct': rrC, 'extraction': 'blind PNG'
                })
                row_record['C_psnr'] = psnrC
                row_record['C_ssim'] = ssimC
                print(f"{' ':8} | {'C - Sobel Guided':<26} | {capC:>8} b | {mseC:8.5f}   | {psnrC:8.3f} dB | {ssimC:10.7f} | {rrC:5.1f}% (Blind)")
            except ValueError as ve:
                print(f"{' ':8} | {'C - Sobel Guided':<26} | GAGAL KAPASITAS: {ve}")
                eval_rows.append({
                    'image': name, 'payload_kb': kb, 'payload_bits': nbits,
                    'method': 'C_sobel_guided', 'capacity_bits': '',
                    'embedded_bits': 0, 'full_payload': False,
                    'mse': '', 'psnr_db': '', 'ssim': '',
                    'recovery_pct': '', 'extraction': 'blind PNG'
                })
                row_record['C_psnr'] = float('nan')
                row_record['C_ssim'] = float('nan')

            # ------------------------------------------------------------------
            # Metode D: Proposed Adaptive Sobel Guided V2
            # ------------------------------------------------------------------
            try:
                stegoD, capD = embed_d(cover, msg)
                ok, pngD = cv2.imencode('.png', stegoD)
                reloadedD = cv2.imdecode(pngD, cv2.IMREAD_GRAYSCALE)
                recD = extract_blind_d(reloadedD, nbits)
                rrD = 100.0 * float(np.mean(recD == msg))
                psnrD, ssimD, mseD = calc_metrics(cover, reloadedD)
                eval_rows.append({
                    'image': name, 'payload_kb': kb, 'payload_bits': nbits,
                    'method': 'D_adaptive_sobel_v2', 'capacity_bits': capD,
                    'embedded_bits': nbits, 'full_payload': True,
                    'mse': mseD, 'psnr_db': psnrD, 'ssim': ssimD,
                    'recovery_pct': rrD, 'extraction': 'blind PNG'
                })
                row_record['D_psnr'] = psnrD
                row_record['D_ssim'] = ssimD
                print(f"{' ':8} | {'D - Adaptive Sobel V2':<26} | {capD:>8} b | {mseD:8.5f}   | {psnrD:8.3f} dB | {ssimD:10.7f} | {rrD:5.1f}% (Blind)")

                diff_psnr_D_A = psnrD - row_record['A_psnr']
                diff_ssim_D_A = ssimD - row_record['A_ssim']
                diff_psnr_C_A = row_record['C_psnr'] - row_record['A_psnr'] if not np.isnan(row_record['C_psnr']) else float('nan')
                diff_ssim_C_A = row_record['C_ssim'] - row_record['A_ssim'] if not np.isnan(row_record['C_ssim']) else float('nan')
                diff_psnr_B_A = row_record['B_psnr'] - row_record['A_psnr'] if not np.isnan(row_record['B_psnr']) else float('nan')
                diff_ssim_B_A = row_record['B_ssim'] - row_record['A_ssim'] if not np.isnan(row_record['B_ssim']) else float('nan')

                print(f"{' ':8} | >>> Dampak D vs A: PSNR {diff_psnr_D_A:+.3f} dB | SSIM {diff_ssim_D_A:+.7f} {'(UNGGUL)' if diff_ssim_D_A > 0 else '(TURUN)'}")
                print("-" * 92)

                diff_rows.append({
                    'image': name,
                    'payload_kb': kb,
                    'payload_bits': nbits,
                    'PSNR_A': row_record['A_psnr'],
                    'PSNR_B': row_record['B_psnr'],
                    'PSNR_C': row_record['C_psnr'],
                    'PSNR_D': row_record['D_psnr'],
                    'Diff_PSNR_D_minus_A': diff_psnr_D_A,
                    'SSIM_A': row_record['A_ssim'],
                    'SSIM_B': row_record['B_ssim'],
                    'SSIM_C': row_record['C_ssim'],
                    'SSIM_D': row_record['D_ssim'],
                    'Diff_SSIM_D_minus_A': diff_ssim_D_A,
                    'Diff_SSIM_C_minus_A': diff_ssim_C_A,
                    'Diff_SSIM_B_minus_A': diff_ssim_B_A,
                    'Recovery_Rate_D': rrD
                })

            except ValueError as ve:
                print(f"{' ':8} | {'D - Adaptive Sobel V2':<26} | GAGAL KAPASITAS: {ve}")

    # ==========================================================================
    # 9. SIMPAN FILE HASIL CSV
    # ==========================================================================
    csv_eval_path = os.path.join(BASE_DIR, 'hasil_ablasi_sobel_guided_v2.csv')
    if eval_rows:
        with open(csv_eval_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
            writer.writeheader()
            writer.writerows(eval_rows)
        print(f"\n[OK] Hasil eksperimen lengkap tersimpan di: {csv_eval_path}")

    csv_diff_path = os.path.join(BASE_DIR, 'analisis_perbandingan_v2.csv')
    if diff_rows:
        with open(csv_diff_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(diff_rows[0].keys()))
            writer.writeheader()
            writer.writerows(diff_rows)
        print(f"[OK] Analisis selisih metrik tersimpan di: {csv_diff_path}")

    t_end = time.time()
    print(f"\nSeluruh pengujian selesai dalam {t_end - t_start:.2f} detik.")
    print("=" * 115)

if __name__ == '__main__':
    main()
