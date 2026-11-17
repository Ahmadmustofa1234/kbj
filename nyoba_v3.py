"""
================================================================================
EKSPERIMEN LENGKAP METODE STEGANOGRAFI MULTI-PARAMETER & SOBEL GUIDED V3
Pengembangan Paper: Enhanced Image Steganography Through Multi-Parameter Texture Analysis
Peneliti          : Senior Python Developer & Peneliti Image Steganography

Metode yang dibandingkan:
  [A] Baseline Makalah Asli   : 3 Parameter + PairStruct (Raster Order, Cover-Assisted)
  [B] Sobel Raster            : 3 Param + Sobel non_med<=1 (Raster Order, Blind PNG)
  [C] Sobel Guided            : 3 Param + Sobel non_med<=1 (Category Guided, Blind PNG)
  [D] Adaptive Sobel Guided V2: 4 Param Sejati + PairStruct (Adaptive Sobel Masking, Blind PNG)
  [E] Proposed Method E       : Adaptive Four-Parameter Sobel + Guided Pixel Selection (Blind PNG)
        - Sobel sebagai parameter ke-4 aktif menentukan kapasitas 0/1/2 bit
        - PairStruct sebagai pembatas kapasitas akhir (6-MSB)
        - Guided Pixel Selection deterministik dari citra stego
        - True Blind Extraction dari berkas PNG lossless (RR = 100.0%)

Ablasi Tambahan untuk Memisahkan Kontribusi Sobel:
  [E_cap_only]   : Kapasitas 4-Parameter Sobel, tetapi Urutan Raster Scan
  [E_route_only] : Kapasitas 3-Parameter Asli, tetapi Urutan Guided Sobel
  [E_full]       : Kapasitas 4-Parameter Sobel + Urutan Guided Sobel
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
# 1. LOOK-UP TABLE (LUT) ENTROPI SHANNON CEPAT
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
# 2. PEMUATAN DATASET CITRA BENCHMARK (ALAMI & MEDIS)
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
# 3. SEED PENELITIAN & URUTAN PENGUJIAN
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
# 4. METODE A: BASELINE ASLI (3-PARAMETER + PAIRSTRUCT, COVER-ASSISTED)
# ==============================================================================
def compute_multi_parameter_capacity(img):
    """Menghitung kapasitas bit per piksel berdasarkan 3 parameter tekstur + PairStruct asli."""
    windows = sliding_window_view(img, (3, 3))
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
        return {'stego': stego_img, 'psnr': float('inf'), 'ssim': 1.0, 'rr': 100.0, 'embedded_bits': 0, 'capacity': int(cap_arr.sum()), 'mse': 0.0}

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
    mse = float(np.mean((cover_img.astype(np.float64) - stego_img.astype(np.float64)) ** 2))
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
    stable = np.bitwise_and(image, np.uint8(0xFC))
    capacity = compute_multi_parameter_capacity(stable).astype(np.uint8)
    gx = cv2.Sobel(stable, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(stable, cv2.CV_64F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)[1:-1, 1:-1].ravel()

    cap = capacity.copy()
    non_medium = (gradient < LOW) | (gradient > HIGH)
    cap[non_medium] = np.minimum(cap[non_medium], 1)
    idx = np.flatnonzero(cap > 0)
    if guided:
        category = np.where(gradient[idx] < LOW, 1,
                            np.where(gradient[idx] <= HIGH, 0, 2))
        idx = idx[np.argsort(category, kind='stable')]
    return cap, idx

def embed_bc(image, message, guided=False):
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
# 6. METODE D: ADAPTIVE SOBEL GUIDED V2 (DARI NYOBA_V2.PY)
# ==============================================================================
def stable_plan_d(image):
    stable = np.bitwise_and(image, np.uint8(0xFC))
    win = sliding_window_view(stable, (3, 3))
    flat_win = win.reshape(-1, 9)

    std_arr = np.std(flat_win, axis=1)
    rng_arr = np.ptp(flat_win, axis=1)
    s = np.sort(flat_win, axis=1)
    diffs = (s[:, 1:] != s[:, :-1])
    masks = diffs.dot(POWERS_OF_TWO)
    entropy_arr = ENTROPY_LUT[masks]

    gx = cv2.Sobel(stable, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(stable, cv2.CV_64F, 0, 1, ksize=3)
    grad_arr = cv2.magnitude(gx, gy)[1:-1, 1:-1].ravel()

    s_low, s_med, s_hi = std_arr < 10, (std_arr >= 10) & (std_arr <= 25), std_arr > 25
    r_low, r_med, r_hi = rng_arr < 20, (rng_arr >= 20) & (rng_arr <= 50), rng_arr > 50
    e_low, e_med, e_hi = entropy_arr < 1.2, (entropy_arr >= 1.2) & (entropy_arr <= 2.2), entropy_arr > 2.2
    g_low, g_med, g_hi = grad_arr < 30, (grad_arr >= 30) & (grad_arr <= 100), grad_arr > 100

    low_cnt = s_low.astype(int) + r_low.astype(int) + e_low.astype(int) + g_low.astype(int)
    med_cnt = s_med.astype(int) + r_med.astype(int) + e_med.astype(int) + g_med.astype(int)
    hi_cnt  = s_hi.astype(int) + r_hi.astype(int) + e_hi.astype(int) + g_hi.astype(int)

    d_mat = np.abs(stable[1:-1, 2:].astype(np.int32) - stable[1:-1, 1:-1].astype(np.int32))
    d_arr = d_mat.ravel()

    mask_2bit = ((med_cnt >= 3) & (hi_cnt == 0) & (d_arr <= 8)) | \
                ((med_cnt == 2) & g_med & (hi_cnt == 0) & (d_arr <= 8))

    mask_1bit = (
        ((low_cnt >= 2) & (hi_cnt == 0)) |
        ((med_cnt >= 2) & (hi_cnt <= 1)) |
        ((hi_cnt == 1) & (med_cnt >= 1)) |
        ((low_cnt >= 1) & (med_cnt >= 1) & (hi_cnt <= 1))
    ) & (hi_cnt < 3)

    eligible = mask_1bit | mask_2bit
    idx_eligible = np.flatnonzero(eligible)

    score_l1 = std_arr[idx_eligible] * 2.0 + grad_arr[idx_eligible]
    order_layer1 = idx_eligible[np.argsort(-score_l1, kind='stable')]

    idx_2bit = np.flatnonzero(mask_2bit)
    score_l2 = std_arr[idx_2bit] * 2.0 + grad_arr[idx_2bit]
    order_layer2 = idx_2bit[np.argsort(-score_l2, kind='stable')]

    total_capacity = len(order_layer1) + len(order_layer2)
    return order_layer1, order_layer2, total_capacity

def embed_d(image, message):
    message = np.asarray(message, dtype=np.uint8)
    order_l1, order_l2, cap = stable_plan_d(image)
    nbits = len(message)
    if nbits > cap:
        raise ValueError(f"Payload {nbits} bit melebihi kapasitas {cap} bit")

    stego = image.copy()
    interior = stego[1:-1, 1:-1].reshape(-1).copy()

    n1 = min(nbits, len(order_l1))
    for i in range(n1):
        idx = order_l1[i]
        interior[idx] = (int(interior[idx]) & 0xFE) | int(message[i])

    if nbits > len(order_l1):
        n2 = nbits - len(order_l1)
        for i in range(n2):
            idx = order_l2[i]
            interior[idx] = (int(interior[idx]) & 0xFD) | (int(message[len(order_l1) + i]) << 1)

    stego[1:-1, 1:-1] = interior.reshape(image.shape[0] - 2, image.shape[1] - 2)
    return stego, cap

def extract_blind_d(stego, nbits):
    order_l1, order_l2, cap = stable_plan_d(stego)
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
# 7. METODE E (PROPOSED): ADAPTIVE 4-PARAMETER SOBEL + GUIDED PIXEL SELECTION
# ==============================================================================
def compute_four_parameter_sobel_model(image):
    """
    Model Komputasi 4-Parameter Metode E pada domain 6-MSB:
    1. Standard Deviation (STD)
    2. Range (R)
    3. Shannon Entropy (H)
    4. Sobel Gradient Magnitude (G)
    
    Menghasilkan:
    - final_cap      : Kapasitas final per piksel (0, 1, atau 2 bit)
    - feat4_bits     : Keputusan alokasi dari 4 fitur tekstur
    - feat3_bits     : Keputusan alokasi dari 3 fitur tekstur (tanpa Sobel)
    - changed_mask   : Mask boolean piksel yang berubah kapasitasnya karena Sobel
    - order_guided   : Urutan guided adaptive masking score (Layer 1 & Layer 2)
    - total_capacity : Total bit yang mampu ditampung
    """
    stable = np.bitwise_and(image, np.uint8(0xFC))
    win = sliding_window_view(stable, (3, 3))
    flat_win = win.reshape(-1, 9)

    std_arr = np.std(flat_win, axis=1)
    rng_arr = np.ptp(flat_win, axis=1)
    s = np.sort(flat_win, axis=1)
    diffs = (s[:, 1:] != s[:, :-1])
    masks = diffs.dot(POWERS_OF_TWO)
    entropy_arr = ENTROPY_LUT[masks]

    gx = cv2.Sobel(stable, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(stable, cv2.CV_64F, 0, 1, ksize=3)
    grad_arr = cv2.magnitude(gx, gy)[1:-1, 1:-1].ravel()

    # Klasifikasi 4 Parameter
    s_low, s_med, s_hi = std_arr < 10, (std_arr >= 10) & (std_arr <= 25), std_arr > 25
    r_low, r_med, r_hi = rng_arr < 20, (rng_arr >= 20) & (rng_arr <= 50), rng_arr > 50
    e_low, e_med, e_hi = entropy_arr < 1.2, (entropy_arr >= 1.2) & (entropy_arr <= 2.2), entropy_arr > 2.2
    g_low, g_med, g_hi = grad_arr < 30, (grad_arr >= 30) & (grad_arr <= 100), grad_arr > 100

    # Konsensus 3 Fitur (untuk mengukur baseline tanpa Sobel)
    low3 = s_low.astype(int) + r_low.astype(int) + e_low.astype(int)
    med3 = s_med.astype(int) + r_med.astype(int) + e_med.astype(int)
    hi3  = s_hi.astype(int) + r_hi.astype(int) + e_hi.astype(int)

    feat3_bits = np.zeros(len(std_arr), dtype=np.int32)
    feat3_bits[low3 == 3] = 1
    feat3_bits[(med3 >= 2) & (hi3 == 0)] = 2
    feat3_bits[hi3 == 1] = 1

    # Konsensus 4 Fitur Sejati (dengan Sobel aktif)
    low4 = low3 + g_low.astype(int)
    med4 = med3 + g_med.astype(int)
    hi4  = hi3  + g_hi.astype(int)

    # --------------------------------------------------------------------------
    # ATURAN KEPUTUSAN 4-PARAMETER METODE E
    # --------------------------------------------------------------------------
    # Case E-2 (Tekstur Kaya): N_med >= 3 dan 0 High, ATAU (N_med == 2 DAN Sobel Med DAN 0 High)
    case_2bit = ((med4 >= 3) & (hi4 == 0)) | \
                ((med4 == 2) & g_med & (hi4 == 0))

    # Case E-1, E-3, E-5 (Halus, Transisi, dan Moderat)
    case_1bit = (
        ((low4 >= 3) & (hi4 == 0)) |
        ((low4 == 2) & (med4 >= 1) & (hi4 <= 1)) |
        ((med4 >= 2) & (hi4 <= 1)) |
        ((hi4 == 1) & (med4 >= 1)) |
        ((hi4 == 2) & ~g_hi & (med4 >= 1))
    )

    feat4_bits = np.zeros(len(std_arr), dtype=np.int32)
    feat4_bits[case_2bit] = 2
    feat4_bits[case_1bit & ~case_2bit] = 1

    # PairStruct Boundary Constraint pada 6-MSB
    d_mat = np.abs(stable[1:-1, 2:].astype(np.int32) - stable[1:-1, 1:-1].astype(np.int32))
    d_arr = d_mat.ravel()

    pair_bits = np.zeros(len(std_arr), dtype=np.int32)
    pair_bits[(std_arr <= 15) & (d_arr <= 4)] = 1
    pair_bits[(std_arr > 15) & (std_arr <= 20) & (d_arr <= 4)] = 2
    pair_bits[(std_arr > 20) & (std_arr <= 30) & (d_arr > 4)] = 2

    # Decision Fusion Akhir:
    final_cap = np.where(
        feat4_bits > 0,
        np.maximum(1, np.minimum(feat4_bits, np.where(pair_bits > 0, pair_bits, 1))),
        0
    ).astype(np.uint8)

    # Piksel yang keputusan kapasitasnya berubah langsung karena Sobel
    changed_mask = (feat3_bits != feat4_bits)

    # Guided Pixel Selection: Adaptive Score S(i) = G(i) + 2*sigma(i) + R(i)
    mask_eligible = (final_cap > 0)
    idx_eligible = np.flatnonzero(mask_eligible)
    score_all = grad_arr[idx_eligible] + 2.0 * std_arr[idx_eligible] + rng_arr[idx_eligible]
    order_layer1 = idx_eligible[np.argsort(-score_all, kind='stable')]

    mask_2bit = (final_cap == 2)
    idx_2bit = np.flatnonzero(mask_2bit)
    score_2b = grad_arr[idx_2bit] + 2.0 * std_arr[idx_2bit] + rng_arr[idx_2bit]
    order_layer2 = idx_2bit[np.argsort(-score_2b, kind='stable')]

    total_capacity = len(order_layer1) + len(order_layer2)

    return {
        'final_cap': final_cap,
        'feat4_bits': feat4_bits,
        'feat3_bits': feat3_bits,
        'changed_mask': changed_mask,
        'order_layer1': order_layer1,
        'order_layer2': order_layer2,
        'total_capacity': total_capacity,
        'grad_arr': grad_arr,
        'std_arr': std_arr,
        'rng_arr': rng_arr
    }

def embed_e(image, message):
    """Penyisipan adaptif Metode E."""
    message = np.asarray(message, dtype=np.uint8)
    model = compute_four_parameter_sobel_model(image)
    order_l1 = model['order_layer1']
    order_l2 = model['order_layer2']
    cap = model['total_capacity']
    nbits = len(message)

    if nbits > cap:
        raise ValueError(f"Payload {nbits} bit melebihi kapasitas {cap} bit")

    stego = image.copy()
    interior = stego[1:-1, 1:-1].reshape(-1).copy()

    # Phase 1: Bit 0 pada Layer 1
    n1 = min(nbits, len(order_l1))
    for i in range(n1):
        idx = order_l1[i]
        interior[idx] = (int(interior[idx]) & 0xFE) | int(message[i])

    # Phase 2: Bit 1 pada Layer 2 jika muatan melebihi piksel Layer 1
    if nbits > len(order_l1):
        n2 = nbits - len(order_l1)
        for i in range(n2):
            idx = order_l2[i]
            interior[idx] = (int(interior[idx]) & 0xFD) | (int(message[len(order_l1) + i]) << 1)

    stego[1:-1, 1:-1] = interior.reshape(image.shape[0] - 2, image.shape[1] - 2)
    return stego, cap, model

def extract_blind_e(stego, nbits):
    """Ekstraksi blind tanpa cover untuk Metode E."""
    model = compute_four_parameter_sobel_model(stego)
    order_l1 = model['order_layer1']
    order_l2 = model['order_layer2']
    cap = model['total_capacity']

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
# 8. VARIAN ABLASI UNTUK ISOLASI KONTRIBUSI SOBEL (KAPASITAS VS URUTAN)
# ==============================================================================
def embed_e_cap_only(image, message):
    """Ablasi 1: Kapasitas 4-Parameter Sobel, tetapi urutan pemindaian RASTER murni."""
    message = np.asarray(message, dtype=np.uint8)
    model = compute_four_parameter_sobel_model(image)
    final_cap = model['final_cap']
    available = int(np.sum(final_cap))
    nbits = len(message)
    if nbits > available:
        raise ValueError(f"Payload {nbits} bit melebihi kapasitas {available} bit")

    stego = image.copy()
    interior = stego[1:-1, 1:-1].reshape(-1).copy()
    nz_idx = np.flatnonzero(final_cap > 0)
    pos = 0
    for idx in nz_idx:
        if pos == nbits:
            break
        n = min(int(final_cap[idx]), nbits - pos)
        if n == 1:
            interior[idx] = (int(interior[idx]) & 0xFE) | int(message[pos])
        else:
            interior[idx] = (int(interior[idx]) & 0xFC) | (int(message[pos]) << 1) | int(message[pos + 1])
        pos += n

    stego[1:-1, 1:-1] = interior.reshape(image.shape[0] - 2, image.shape[1] - 2)
    return stego, available

def extract_blind_e_cap_only(stego, nbits):
    """Ekstraksi blind untuk ablasi E_cap_only."""
    model = compute_four_parameter_sobel_model(stego)
    final_cap = model['final_cap']
    interior = stego[1:-1, 1:-1].ravel()
    bits = np.empty(nbits, dtype=np.uint8)
    nz_idx = np.flatnonzero(final_cap > 0)
    pos = 0
    for idx in nz_idx:
        if pos == nbits:
            break
        n = min(int(final_cap[idx]), nbits - pos)
        val = int(interior[idx])
        if n == 1:
            bits[pos] = val & 1
        else:
            bits[pos] = (val >> 1) & 1
            bits[pos + 1] = val & 1
        pos += n
    return bits

def embed_e_route_only(image, message):
    """Ablasi 2: Kapasitas 3-Parameter Asli (6-MSB), tetapi urutan guided Sobel."""
    message = np.asarray(message, dtype=np.uint8)
    stable = np.bitwise_and(image, np.uint8(0xFC))
    cap_3p = compute_multi_parameter_capacity(stable).astype(np.uint8)
    available = int(np.sum(cap_3p))
    nbits = len(message)
    if nbits > available:
        raise ValueError(f"Payload {nbits} bit melebihi kapasitas {available} bit")

    gx = cv2.Sobel(stable, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(stable, cv2.CV_64F, 0, 1, ksize=3)
    grad_arr = cv2.magnitude(gx, gy)[1:-1, 1:-1].ravel()

    idx_nz = np.flatnonzero(cap_3p > 0)
    score = grad_arr[idx_nz]
    order = idx_nz[np.argsort(-score, kind='stable')]

    stego = image.copy()
    interior = stego[1:-1, 1:-1].reshape(-1).copy()
    pos = 0
    for idx in order:
        if pos == nbits:
            break
        n = min(int(cap_3p[idx]), nbits - pos)
        if n == 1:
            interior[idx] = (int(interior[idx]) & 0xFE) | int(message[pos])
        else:
            interior[idx] = (int(interior[idx]) & 0xFC) | (int(message[pos]) << 1) | int(message[pos + 1])
        pos += n

    stego[1:-1, 1:-1] = interior.reshape(image.shape[0] - 2, image.shape[1] - 2)
    return stego, available

def extract_blind_e_route_only(stego, nbits):
    """Ekstraksi blind untuk ablasi E_route_only."""
    stable = np.bitwise_and(stego, np.uint8(0xFC))
    cap_3p = compute_multi_parameter_capacity(stable).astype(np.uint8)
    gx = cv2.Sobel(stable, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(stable, cv2.CV_64F, 0, 1, ksize=3)
    grad_arr = cv2.magnitude(gx, gy)[1:-1, 1:-1].ravel()

    idx_nz = np.flatnonzero(cap_3p > 0)
    score = grad_arr[idx_nz]
    order = idx_nz[np.argsort(-score, kind='stable')]

    interior = stego[1:-1, 1:-1].ravel()
    bits = np.empty(nbits, dtype=np.uint8)
    pos = 0
    for idx in order:
        if pos == nbits:
            break
        n = min(int(cap_3p[idx]), nbits - pos)
        val = int(interior[idx])
        if n == 1:
            bits[pos] = val & 1
        else:
            bits[pos] = (val >> 1) & 1
            bits[pos + 1] = val & 1
        pos += n
    return bits

# ==============================================================================
# 9. METRIK EVALUASI KUALITAS
# ==============================================================================
def calc_metrics(cover, stego):
    mse = float(np.mean((cover.astype(np.float64) - stego.astype(np.float64)) ** 2))
    psnr = float('inf') if mse == 0 else 10.0 * np.log10((255.0 ** 2) / mse)
    ssim_val = float(ssim(cover, stego, data_range=255))
    return psnr, ssim_val, mse

# ==============================================================================
# 10. EKSEKUSI UTAMA (MAIN EXPERIMENTAL HARNESS)
# ==============================================================================
def main():
    t_global_start = time.time()
    print("=" * 120)
    print("HASIL EKSPERIMEN LENGKAP METODE STEGANOGRAFI A, B, C, D, E & ABLASI SOBEL")
    print("  [A] Baseline Makalah Asli   : 3 Parameter + PairStruct (Raster Order, Cover-Assisted)")
    print("  [B] Sobel Raster            : 3 Param + Sobel non_med<=1 (Raster Order, Blind PNG)")
    print("  [C] Sobel Guided            : 3 Param + Sobel non_med<=1 (Category Guided, Blind PNG)")
    print("  [D] Adaptive Sobel Guided V2: 4 Param Sejati + PairStruct (Adaptive Sobel Masking, Blind PNG)")
    print("  [E] Proposed Method E       : 4-Parameter Sobel Capacity Decision + Guided Selection (Blind PNG)")
    print("=" * 120)

    images = load_all_dataset()
    avail = [k for k in ORDER if k in images]
    print(f"\n-> Citra benchmark termuat ({len(avail)}): {avail}\n")

    eval_v3_rows = []
    sobel_contrib_rows = []

    for name in ORDER:
        if name not in images:
            print(f"SKIP {name}: berkas citra tidak ditemukan")
            continue

        cover = images[name]
        print(f"\n{'=' * 120}")
        print(f"CITRA: {name.upper()}")
        print(f"{'=' * 120}")
        print(f"{'Payload':<8} | {'Metode':<27} | {'Kapasitas':<10} | {'MSE':<10} | {'PSNR (dB)':<10} | {'SSIM':<11} | {'RR %':<7} | {'Waktu':<8}")
        print("-" * 120)

        for kb in PAYLOAD_KB:
            nbits = kb * 1000
            seed = PAPER_MATCHED_SEEDS[name][kb]
            msg = np.random.default_rng(seed).integers(0, 2, size=nbits, dtype=np.uint8)

            record = {'image': name, 'payload_kb': kb, 'payload_bits': nbits}

            # ------------------------------------------------------------------
            # [A] Baseline
            # ------------------------------------------------------------------
            t0 = time.time()
            resA = run_multi_parameter_proposed(cover, msg)
            t_A = time.time() - t0
            record['A_psnr'], record['A_ssim'], record['A_cap'] = resA['psnr'], resA['ssim'], resA['capacity']
            eval_v3_rows.append({
                'image': name, 'payload_kb': kb, 'payload_bits': nbits, 'method': 'A_baseline_original',
                'capacity_bits': resA['capacity'], 'embedded_bits': resA['embedded_bits'],
                'full_payload': resA['embedded_bits'] == nbits, 'mse': resA['mse'], 'psnr_db': resA['psnr'],
                'ssim': resA['ssim'], 'recovery_pct': resA['rr'], 'extraction': 'cover-assisted',
                'time_sec': t_A
            })
            print(f"{kb:>3} kb   | {'A - Baseline Makalah':<27} | {resA['capacity']:>8} b | {resA['mse']:8.5f}   | {resA['psnr']:8.3f} dB | {resA['ssim']:10.7f} | {resA['rr']:5.1f}% | {t_A:5.3f}s")

            # ------------------------------------------------------------------
            # [B] Sobel Raster
            # ------------------------------------------------------------------
            t0 = time.time()
            try:
                stegoB, capB = embed_bc(cover, msg, guided=False)
                ok, pngB = cv2.imencode('.png', stegoB)
                reloadedB = cv2.imdecode(pngB, cv2.IMREAD_GRAYSCALE)
                recB = extract_blind_bc(reloadedB, nbits, guided=False)
                t_B = time.time() - t0
                rrB = 100.0 * float(np.mean(recB == msg))
                psnrB, ssimB, mseB = calc_metrics(cover, reloadedB)
                record['B_psnr'], record['B_ssim'], record['B_cap'] = psnrB, ssimB, capB
                eval_v3_rows.append({
                    'image': name, 'payload_kb': kb, 'payload_bits': nbits, 'method': 'B_sobel_raster',
                    'capacity_bits': capB, 'embedded_bits': nbits, 'full_payload': True,
                    'mse': mseB, 'psnr_db': psnrB, 'ssim': ssimB, 'recovery_pct': rrB,
                    'extraction': 'blind PNG', 'time_sec': t_B
                })
                print(f"{' ':8} | {'B - Sobel Raster':<27} | {capB:>8} b | {mseB:8.5f}   | {psnrB:8.3f} dB | {ssimB:10.7f} | {rrB:5.1f}% | {t_B:5.3f}s (Blind)")
            except ValueError as ve:
                t_B = time.time() - t0
                record['B_psnr'], record['B_ssim'], record['B_cap'] = float('nan'), float('nan'), 0
                eval_v3_rows.append({
                    'image': name, 'payload_kb': kb, 'payload_bits': nbits, 'method': 'B_sobel_raster',
                    'capacity_bits': '', 'embedded_bits': 0, 'full_payload': False,
                    'mse': '', 'psnr_db': '', 'ssim': '', 'recovery_pct': '',
                    'extraction': 'blind PNG', 'time_sec': t_B
                })
                print(f"{' ':8} | {'B - Sobel Raster':<27} | GAGAL KAPASITAS: {ve}")

            # ------------------------------------------------------------------
            # [C] Sobel Guided
            # ------------------------------------------------------------------
            t0 = time.time()
            try:
                stegoC, capC = embed_bc(cover, msg, guided=True)
                ok, pngC = cv2.imencode('.png', stegoC)
                reloadedC = cv2.imdecode(pngC, cv2.IMREAD_GRAYSCALE)
                recC = extract_blind_bc(reloadedC, nbits, guided=True)
                t_C = time.time() - t0
                rrC = 100.0 * float(np.mean(recC == msg))
                psnrC, ssimC, mseC = calc_metrics(cover, reloadedC)
                record['C_psnr'], record['C_ssim'], record['C_cap'] = psnrC, ssimC, capC
                eval_v3_rows.append({
                    'image': name, 'payload_kb': kb, 'payload_bits': nbits, 'method': 'C_sobel_guided',
                    'capacity_bits': capC, 'embedded_bits': nbits, 'full_payload': True,
                    'mse': mseC, 'psnr_db': psnrC, 'ssim': ssimC, 'recovery_pct': rrC,
                    'extraction': 'blind PNG', 'time_sec': t_C
                })
                print(f"{' ':8} | {'C - Sobel Guided':<27} | {capC:>8} b | {mseC:8.5f}   | {psnrC:8.3f} dB | {ssimC:10.7f} | {rrC:5.1f}% | {t_C:5.3f}s (Blind)")
            except ValueError as ve:
                t_C = time.time() - t0
                record['C_psnr'], record['C_ssim'], record['C_cap'] = float('nan'), float('nan'), 0
                eval_v3_rows.append({
                    'image': name, 'payload_kb': kb, 'payload_bits': nbits, 'method': 'C_sobel_guided',
                    'capacity_bits': '', 'embedded_bits': 0, 'full_payload': False,
                    'mse': '', 'psnr_db': '', 'ssim': '', 'recovery_pct': '',
                    'extraction': 'blind PNG', 'time_sec': t_C
                })
                print(f"{' ':8} | {'C - Sobel Guided':<27} | GAGAL KAPASITAS: {ve}")

            # ------------------------------------------------------------------
            # [D] Adaptive Sobel Guided V2
            # ------------------------------------------------------------------
            t0 = time.time()
            stegoD, capD = embed_d(cover, msg)
            ok, pngD = cv2.imencode('.png', stegoD)
            reloadedD = cv2.imdecode(pngD, cv2.IMREAD_GRAYSCALE)
            recD = extract_blind_d(reloadedD, nbits)
            t_D = time.time() - t0
            rrD = 100.0 * float(np.mean(recD == msg))
            psnrD, ssimD, mseD = calc_metrics(cover, reloadedD)
            record['D_psnr'], record['D_ssim'], record['D_cap'] = psnrD, ssimD, capD
            eval_v3_rows.append({
                'image': name, 'payload_kb': kb, 'payload_bits': nbits, 'method': 'D_adaptive_sobel_v2',
                'capacity_bits': capD, 'embedded_bits': nbits, 'full_payload': True,
                'mse': mseD, 'psnr_db': psnrD, 'ssim': ssimD, 'recovery_pct': rrD,
                'extraction': 'blind PNG', 'time_sec': t_D
            })
            print(f"{' ':8} | {'D - Adaptive Sobel V2':<27} | {capD:>8} b | {mseD:8.5f}   | {psnrD:8.3f} dB | {ssimD:10.7f} | {rrD:5.1f}% | {t_D:5.3f}s (Blind)")

            # ------------------------------------------------------------------
            # [E] Proposed Method E: 4-Parameter Sobel Capacity + Guided Selection
            # ------------------------------------------------------------------
            t0 = time.time()
            stegoE, capE, modelE = embed_e(cover, msg)
            ok, pngE = cv2.imencode('.png', stegoE)
            reloadedE = cv2.imdecode(pngE, cv2.IMREAD_GRAYSCALE)
            recE = extract_blind_e(reloadedE, nbits)
            t_E = time.time() - t0
            rrE = 100.0 * float(np.mean(recE == msg))
            psnrE, ssimE, mseE = calc_metrics(cover, reloadedE)
            record['E_psnr'], record['E_ssim'], record['E_cap'] = psnrE, ssimE, capE
            eval_v3_rows.append({
                'image': name, 'payload_kb': kb, 'payload_bits': nbits, 'method': 'E_four_param_sobel_guided',
                'capacity_bits': capE, 'embedded_bits': nbits, 'full_payload': True,
                'mse': mseE, 'psnr_db': psnrE, 'ssim': ssimE, 'recovery_pct': rrE,
                'extraction': 'blind PNG', 'time_sec': t_E
            })
            print(f"{' ':8} | {'E - Proposed 4-Param Sobel':<27} | {capE:>8} b | {mseE:8.5f}   | {psnrE:8.3f} dB | {ssimE:10.7f} | {rrE:5.1f}% | {t_E:5.3f}s (Blind)")

            # ------------------------------------------------------------------
            # ABLASI TAMBAHAN: ISOLASI KONTRIBUSI SOBEL
            # ------------------------------------------------------------------
            # Ablasi 1: E_cap_only (Kapasitas 4-parameter, urutan raster scan)
            t0 = time.time()
            stego_ab1, cap_ab1 = embed_e_cap_only(cover, msg)
            ok, png_ab1 = cv2.imencode('.png', stego_ab1)
            reloaded_ab1 = cv2.imdecode(png_ab1, cv2.IMREAD_GRAYSCALE)
            rec_ab1 = extract_blind_e_cap_only(reloaded_ab1, nbits)
            t_ab1 = time.time() - t0
            rr_ab1 = 100.0 * float(np.mean(rec_ab1 == msg))
            psnr_ab1, ssim_ab1, mse_ab1 = calc_metrics(cover, reloaded_ab1)
            eval_v3_rows.append({
                'image': name, 'payload_kb': kb, 'payload_bits': nbits, 'method': 'E_ablasi_cap_only_raster',
                'capacity_bits': cap_ab1, 'embedded_bits': nbits, 'full_payload': True,
                'mse': mse_ab1, 'psnr_db': psnr_ab1, 'ssim': ssim_ab1, 'recovery_pct': rr_ab1,
                'extraction': 'blind PNG', 'time_sec': t_ab1
            })

            # Ablasi 2: E_route_only (Kapasitas 3-parameter asli, urutan guided Sobel)
            t0 = time.time()
            try:
                stego_ab2, cap_ab2 = embed_e_route_only(cover, msg)
                ok, png_ab2 = cv2.imencode('.png', stego_ab2)
                reloaded_ab2 = cv2.imdecode(png_ab2, cv2.IMREAD_GRAYSCALE)
                rec_ab2 = extract_blind_e_route_only(reloaded_ab2, nbits)
                t_ab2 = time.time() - t0
                rr_ab2 = 100.0 * float(np.mean(rec_ab2 == msg))
                psnr_ab2, ssim_ab2, mse_ab2 = calc_metrics(cover, reloaded_ab2)
                eval_v3_rows.append({
                    'image': name, 'payload_kb': kb, 'payload_bits': nbits, 'method': 'E_ablasi_route_only_guided',
                    'capacity_bits': cap_ab2, 'embedded_bits': nbits, 'full_payload': True,
                    'mse': mse_ab2, 'psnr_db': psnr_ab2, 'ssim': ssim_ab2, 'recovery_pct': rr_ab2,
                    'extraction': 'blind PNG', 'time_sec': t_ab2
                })
            except ValueError as ve:
                psnr_ab2, ssim_ab2, mse_ab2, rr_ab2, cap_ab2, t_ab2 = float('nan'), float('nan'), float('nan'), float('nan'), 0, 0.0

            # Hitung persentase piksel yang berubah kapasitasnya karena Sobel
            sobel_changed_pct = 100.0 * float(np.mean(modelE['changed_mask']))
            up_2b = int(np.sum((modelE['feat3_bits'] < 2) & (modelE['feat4_bits'] == 2)))
            up_1b = int(np.sum((modelE['feat3_bits'] == 0) & (modelE['feat4_bits'] == 1)))
            down_0b = int(np.sum((modelE['feat3_bits'] > 0) & (modelE['feat4_bits'] == 0)))

            diff_ssim_E_A = ssimE - record['A_ssim']
            diff_psnr_E_A = psnrE - record['A_psnr']
            diff_ssim_E_D = ssimE - record['D_ssim']

            print(f"{' ':8} | >>> Dampak E vs A: PSNR {diff_psnr_E_A:+.3f} dB | SSIM {diff_ssim_E_A:+.7f} {'(UNGGUL)' if diff_ssim_E_A > 0 else '(TURUN)'} | Piksel Berubah Sobel: {sobel_changed_pct:.2f}%")
            print("-" * 120)

            sobel_contrib_rows.append({
                'image': name,
                'payload_kb': kb,
                'payload_bits': nbits,
                'sobel_changed_pct': sobel_changed_pct,
                'pixels_promoted_2bit': up_2b,
                'pixels_promoted_1bit': up_1b,
                'pixels_protected_0bit': down_0b,
                'Cap_A': record['A_cap'],
                'Cap_E': capE,
                'PSNR_A': record['A_psnr'],
                'PSNR_B': record['B_psnr'],
                'PSNR_C': record['C_psnr'],
                'PSNR_D': record['D_psnr'],
                'PSNR_E': psnrE,
                'PSNR_E_cap_only': psnr_ab1,
                'PSNR_E_route_only': psnr_ab2,
                'Diff_PSNR_E_minus_A': diff_psnr_E_A,
                'SSIM_A': record['A_ssim'],
                'SSIM_B': record['B_ssim'],
                'SSIM_C': record['C_ssim'],
                'SSIM_D': record['D_ssim'],
                'SSIM_E': ssimE,
                'SSIM_E_cap_only': ssim_ab1,
                'SSIM_E_route_only': ssim_ab2,
                'Diff_SSIM_E_minus_A': diff_ssim_E_A,
                'Diff_SSIM_E_minus_D': diff_ssim_E_D,
                'Recovery_Rate_E': rrE,
                'Comp_Time_E_sec': t_E
            })

    # ==========================================================================
    # 11. MENULIS BERKAS CSV KELUARAN
    # ==========================================================================
    csv_eval_path = os.path.join(BASE_DIR, 'hasil_ablasi_v3.csv')
    if eval_v3_rows:
        with open(csv_eval_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(eval_v3_rows[0].keys()))
            writer.writeheader()
            writer.writerows(eval_v3_rows)
        print(f"\n[OK] Hasil eksperimen komparasi V3 tersimpan: {csv_eval_path}")

    csv_contrib_path = os.path.join(BASE_DIR, 'analisis_kontribusi_sobel.csv')
    if sobel_contrib_rows:
        with open(csv_contrib_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(sobel_contrib_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sobel_contrib_rows)
        print(f"[OK] Analisis kontribusi Sobel tersimpan: {csv_contrib_path}")

    t_global_end = time.time()
    print(f"\nSeluruh pengujian komparasi selesai dalam {t_global_end - t_global_start:.2f} detik.")
    print("=" * 120)

if __name__ == '__main__':
    main()
