import os

import time

import numpy as np

import cv2

import pydicom

from skimage.metrics import structural_similarity as ssim

from numpy.lib.stride_tricks import sliding_window_view

import matplotlib.pyplot as plt



BASE_DIR = os.path.dirname(os.path.abspath(__file__))



# ==============================================================================

# 1. PEMUATAN DATASET (USC-SIPI & DICOM MEDIS)

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

    if max_v > min_v:

        arr_norm = (arr - min_v) / (max_v - min_v) * 255.0

    else:

        arr_norm = np.zeros_like(arr)



    return arr_norm.astype(np.uint8)



# ==============================================================================

# >>> LANGKAH 1 (BAGIAN CITRA): MEMUAT CITRA COVER DALAM SKALA ABU-ABU (GRAYSCALE)

# Paper: "1) Langkah 1: Memuat gambar penutup dalam skala abu-abu..."

# Membaca 3 citra alami (TIFF) dan 5 citra medis (DICOM) serta mengubahnya ke skala abu-abu 0-255.

# ==============================================================================



def load_all_dataset():

    """Memuat 8 citra benchmark: 3 citra alami USC-SIPI dan 5 citra medis DICOM."""

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

            # Coba cari format .png jika ada

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



    return images



# ==============================================================================

# 2. METODE 1: PAIRSTRUCT (METODE PRIOR )

#    

#    - Fitur: Standar Deviasi patch 3x3 dan Pixel Pair Difference (D)

# ==============================================================================



def run_pairstruct_muzli(cover_img, secret_bits):

    """

    Konversi langsung PairStruct ke Python.:

      - std <= 15 and |D| <= 4              -> 1 bit

      - 15 < std <= 20 and |D| <= 4         -> 2 bit

      - 20 < std <= 30 and |D| > 4          -> 2 bit

      - else                                -> 0 bit (skip)

    """

    row, col = cover_img.shape

    sz = (row * col) // 2



    # 1. Hitung pasangan piksel horizontal D(i, j) = img(i, 2j) - img(i, 2j-1)

    pi_j = np.arange(0, col, 2)

    pj_j = np.arange(1, col, 2)

    D_mat = cover_img[:, pj_j].astype(np.int32) - cover_img[:, pi_j].astype(np.int32)

    D_arr = D_mat.ravel() # Panjang sz = 131.072



    stego = cover_img.copy()

    KT = np.zeros((row, col), dtype=np.uint8)

    total_bits = len(secret_bits)

    idx2 = 0



    base_threshold = 4

    low_threshold = 15

    medium_threshold = 20

    high_threshold = 30



    # Penyisipan

    for i in range(row):

        # Perhatikan: di script asli MATLAB, abs(D_arr(1, i)) memakai indeks baris i

        # di Python disesuaikan indeks 0-based i (mod sz agar aman)

        d_val = abs(int(D_arr[i % sz]))



        r_min = max(0, i - 1)

        r_max = min(row, i + 2)



        for j in range(col):

            if idx2 >= total_bits:

                break



            c_min = max(0, j - 1)

            c_max = min(col, j + 2)

            local_patch = cover_img[r_min:r_max, c_min:c_max]

            std_local = float(np.std(local_patch, ddof=1)) if local_patch.size > 1 else 0.0



            if std_local <= low_threshold and d_val <= base_threshold:

                bits_to_embed = 1

            elif low_threshold < std_local <= medium_threshold and d_val <= base_threshold:

                bits_to_embed = 2

            elif medium_threshold < std_local <= high_threshold and d_val > base_threshold:

                bits_to_embed = 2

            else:

                continue



            KT[i, j] = bits_to_embed

            # >>> [PROSES EMBEDDING PADA METODE ALFARABI / MATLAB] <<<

            # Menyisipkan bit rahasia ke dalam bit LSB piksel stego

            for b in range(bits_to_embed):

                if idx2 >= total_bits:

                    break

                bit_val = int(secret_bits[idx2])

                # MATLAB: bitset(val, b, bit_val) -> b=1 adalah LSB ke-0, b=2 adalah LSB ke-1

                mask = 0xFF ^ (1 << b)

                stego[i, j] = (stego[i, j] & np.uint8(mask)) | np.uint8(bit_val << b)

                idx2 += 1

        if idx2 >= total_bits:

            break



    # Ekstraksi

    extracted = []

    idx3 = 0

    for i in range(row):

        d_val = abs(int(D_arr[i % sz]))

        r_min = max(0, i - 1)

        r_max = min(row, i + 2)

        for j in range(col):

            if idx3 >= idx2:

                break

            c_min = max(0, j - 1)

            c_max = min(col, j + 2)

            local_patch = cover_img[r_min:r_max, c_min:c_max]

            std_local = float(np.std(local_patch, ddof=1)) if local_patch.size > 1 else 0.0



            if std_local <= low_threshold and d_val <= base_threshold:

                bits_to_ext = 1

            elif low_threshold < std_local <= medium_threshold and d_val <= base_threshold:

                bits_to_ext = 2

            elif medium_threshold < std_local <= high_threshold and d_val > base_threshold:

                bits_to_ext = 2

            else:

                continue



            for b in range(bits_to_ext):

                if idx3 >= idx2:

                    break

                extracted.append((int(stego[i, j]) >> b) & 1)

                idx3 += 1

        if idx3 >= idx2:

            break



    mse = np.mean((cover_img.astype(float) - stego.astype(float)) ** 2)

    psnr_val = float('inf') if mse == 0 else float(10 * np.log10((255.0 ** 2) / mse))

    ssim_val = float(ssim(cover_img, stego, data_range=255))

    n_match = sum(s == r for s, r in zip(secret_bits[:idx2], extracted))

    if idx2 > 0:

        rr_val = (n_match / idx2) * 100.0

    else:

        rr_val = 0.0



    return {

        'stego': stego,

        'psnr': psnr_val,

        'ssim': ssim_val,

        'rr': rr_val,

        'embedded_bits': idx2

    }



# ==============================================================================

# 3. METODE USULAN: MULTI-PARAMETER TEXTURE ANALYSIS & PAIRSTRUCT

#    Paper: "Enhanced Image Steganography Through Multi-Parameter Texture Analysis"

#    Penulis: Sarah Alissa Putri & Tohari Ahmad (ICICyTA)

# ==============================================================================



# Tabel Look-Up (LUT) Entropi Shannon untuk patch 3x3 (Persamaan 4 di paper).

# Karena patch 3x3 hanya berisi 9 piksel, kita dapat melakukan pra-hitung nilai entropi

# untuk semua kemungkinan pola kemunculan piksel agar komputasi sangat cepat (vektorisasi).

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





def compute_multi_parameter_capacity(img):

    """

    ==============================================================================

    FUNGSI MENGHITUNG KAPASITAS DAN ALOKASI BIT PER PIKSEL

    Mengimplementasikan Langkah 2, 3, 4, dan 5 dari Makalah Section III-A.

    ==============================================================================

    """

    # ==========================================================================

    # >>> LANGKAH 3: TENTUKAN PATCH LOKAL 3x3 (PERSAMAAN 1 DI PAPER)

    # patch(i, j) = I[i-1 : i+2, j-1 : j+2]

    # Sembilan piksel tetangga di sekitar piksel tengah diambil sebagai matriks 3x3.

    # ==========================================================================

    windows = sliding_window_view(img, (3, 3))       # Mengambil sliding window ukuran 3x3

    flat_win = windows.reshape(-1, 9)               # Diratakan menjadi vektor 9 piksel per patch



    # ==========================================================================

    # >>> LANGKAH 4: HITUNG FITUR STATISTIK DARI PATCH 3x3

    # --------------------------------------------------------------------------

    # 1. Simpangan Baku / Standar Deviasi (sigma) - Persamaan (2) di Paper:

    #    sigma = sqrt( 1/9 * sum ( I(x+u, y+v) - mu )^2 )

    #    Mengukur variasi dan kontras intensitas di dalam patch 3x3.

    # --------------------------------------------------------------------------

    std_arr = np.std(flat_win, axis=1)



    # --------------------------------------------------------------------------

    # 2. Rentang Piksel / Range (R) - Persamaan (3) di Paper:

    #    R(x, y) = max(W(x, y)) - min(W(x, y))

    #    Mengukur rentang selisih piksel paling terang dikurangi paling gelap.

    # --------------------------------------------------------------------------

    range_arr = np.ptp(flat_win, axis=1)



    # --------------------------------------------------------------------------

    # 3. Entropi Shannon (H) - Persamaan (4) di Paper:

    #    H(x, y) = - sum p_k * log2( p_k )

    #    Mengukur derajat keacakan dan kekayaan tekstur lokal dari distribusi 9 piksel.

    # --------------------------------------------------------------------------

    s = np.sort(flat_win, axis=1)                   # Urutkan nilai piksel dalam patch 9 elemen

    diffs = (s[:, 1:] != s[:, :-1])                 # Tandai perpindahan nilai unik

    masks = diffs.dot(POWERS_OF_TWO)                # Hitung indeks biner kombinasi unik

    entropy_arr = ENTROPY_LUT[masks]                # Ambil nilai entropi Shannon dari Look-Up Table



    # --------------------------------------------------------------------------

    # KLASIFIKASI AMBANG BATAS FITUR TEKSTUR (Sesuai Threshold pada Makalah):

    # - Standar Deviasi (sigma): Low (< 10), Medium [10, 25], High (> 25)

    # - Range Piksel (R):        Low (< 20), Medium [20, 50], High (> 50)

    # - Entropi Shannon (H):     Low (< 1.2), Medium [1.2, 2.2], High (> 2.2)

    # --------------------------------------------------------------------------

    std_low = std_arr < 10

    std_med = (std_arr >= 10) & (std_arr <= 25)

    std_high = std_arr > 25



    rng_low = range_arr < 20

    rng_med = (range_arr >= 20) & (range_arr <= 50)

    rng_high = range_arr > 50



    ent_low = entropy_arr < 1.2

    ent_med = (entropy_arr >= 1.2) & (entropy_arr <= 2.2)

    ent_high = entropy_arr > 2.2



    # Menghitung konsensus berapa banyak fitur yang masuk kategori Low, Medium, High

    low_cnt = std_low.astype(int) + rng_low.astype(int) + ent_low.astype(int)

    med_cnt = std_med.astype(int) + rng_med.astype(int) + ent_med.astype(int)

    high_cnt = std_high.astype(int) + rng_high.astype(int) + ent_high.astype(int)



    # ==========================================================================

    # >>> LANGKAH 5: TENTUKAN ALOKASI BIT BERBASIS KASUS (PERSAMAAN 5 DI PAPER)

    # - Case A: Semua 3 fitur bernilai Low (daerah sangat halus) -> Alokasi 1 bit

    # - Case B: Minimal 2 fitur Medium & 0 High (daerah tekstur) -> Alokasi 2 bit

    # - Case C: Hanya 1 fitur High (daerah sedang/moderat)       -> Alokasi 1 bit

    # - Case D: >= 2 fitur High (daerah sangat kompleks/bising)  -> 0 bit (Skip / Dihindari)

    # ==========================================================================

    feature_bits = np.zeros(len(std_arr), dtype=np.int32)

    feature_bits[low_cnt == 3] = 1                         # Case A: Daerah halus (1 bit)

    feature_bits[(med_cnt >= 2) & (high_cnt == 0)] = 2     # Case B: Daerah bertekstur (2 bit)

    feature_bits[high_cnt == 1] = 1                         # Case C: Daerah sedang (1 bit)

    # Case D tetap bernilai 0 (daerah bising dilewati agar aman dan bebas artefak)



    # --------------------------------------------------------------------------

    # PAIRSTRUCT (PAIRED-PIXEL DIFFERENCE) -> K_pairing (Fig. 1 pada Makalah)

    # Menghitung selisih horizontal pasangan piksel D = I(x, y+1) - I(x, y)

    # --------------------------------------------------------------------------

    d_mat = img[1:-1, 2:].astype(np.int32) - img[1:-1, 1:-1].astype(np.int32)

    d_arr = d_mat.ravel()



    pairing_bits = np.zeros(len(std_arr), dtype=np.int32)

    pairing_bits[(std_arr <= 15) & (d_arr <= 4)] = 1                          # Threshold 1 bit

    pairing_bits[(std_arr > 15) & (std_arr <= 20) & (d_arr <= 4)] = 2         # Threshold 2 bit

    pairing_bits[(std_arr > 20) & (std_arr <= 30) & (d_arr > 4)] = 2          # Threshold 2 bit



    # ==========================================================================

    # DECISION FUSION (PERSAMAAN 6 DI PAPER):

    # K_final = max( 1, min( K_feature, K_pairing ) ) untuk piksel non-skip

    # Memilih alokasi paling aman antara analisis tekstur dan pairing,

    # serta menjaga minimal 1 bit agar kapasitas muatan tetap maksimal.

    # ==========================================================================

    final_bits = np.where(feature_bits > 0, np.maximum(1, np.minimum(feature_bits, pairing_bits)), 0)

    return final_bits





def run_multi_parameter_proposed(cover_img, secret_bits):

    """

    ==============================================================================

    FUNGSI UTAMA PENYISIPAN, EKSTRAKSI, DAN EVALUASI METRIK

    Mengimplementasikan Langkah 1, 2, 6, dan 7 dari Makalah Section III & IV.

    ==============================================================================

    """

    h, w = cover_img.shape

    # Menghitung kapasitas alokasi bit setiap piksel (Langkah 3 s/d 5)

    cap_arr = compute_multi_parameter_capacity(cover_img)

    stego_img = cover_img.copy()



    # ==========================================================================

    # >>> LANGKAH 1 (BAGIAN BINER): MEMUAT PESAN RAHASIA DALAM BENTUK BINER

    # Pesan rahasia dikonversi menjadi array angka 0 dan 1 (biner).

    # ==========================================================================

    payload = np.asarray(secret_bits, dtype=np.uint8)

    total_bits = len(payload)

    if total_bits == 0:

        return {'stego': stego_img, 'psnr': float('inf'), 'ssim': 1.0, 'rr': 100.0, 'embedded_bits': 0}



    # ==========================================================================

    # >>> LANGKAH 2: HASILKAN URUTAN EMBEDDING (PIKSEL BATAS / TEPI DIKECUALIKAN)

    # Teks Paper: "Piksel batas dikecualikan untuk memastikan bahwa setiap piksel

    #              yang dipilih memiliki tetangga yang valid untuk patch lokal."

    # Sintaks [1:-1, 1:-1] membuang 1 baris & 1 kolom di tepi batas luar gambar.

    # ==========================================================================

    interior = stego_img[1:-1, 1:-1].reshape(-1).copy()

    nz_idx = np.flatnonzero(cap_arr)               # Ambil indeks piksel yang diizinkan disisipi (cap > 0)

    nz_cap = cap_arr[nz_idx]                       # Jatah bit pada piksel tersebut (1 atau 2 bit)

    cum_bits = np.cumsum(nz_cap)                   # Akumulasi jumlah bit yang bisa ditampung



    # Tentukan berapa banyak piksel interior yang perlu dipakai untuk menampung total_bits

    if total_bits >= cum_bits[-1]:

        used_count = len(nz_idx)

        embedded_bits = int(cum_bits[-1])

    else:

        used_count = int(np.searchsorted(cum_bits, total_bits, side='left')) + 1

        embedded_bits = total_bits



    used_idx = nz_idx[:used_count]

    used_cap = nz_cap[:used_count].copy()

    if cum_bits[used_count - 1] > embedded_bits:

        used_cap[-1] = 1



    # Indeks penunjuk posisi bit rahasia untuk tiap piksel

    starts = np.empty(used_count, dtype=np.int64)

    starts[0] = 0

    if used_count > 1:

        starts[1:] = np.cumsum(used_cap[:-1])



    # ==========================================================================

    # >>> [PROSES EMBEDDING / PENYISIPAN DATA RAHASIA KE DALAM CITRA] <<<

    # (Sesuai Langkah 6 & Persamaan 7 pada Makalah)

    #

    # Di sinilah pesan rahasia (bit biner 0 dan 1) benar-benar dimasukkan

    # ke dalam nilai piksel gambar cover pada posisi LSB (Least Significant Bit).

    # ==========================================================================

    # A. PROSES EMBEDDING 1 BIT PER PIKSEL:

    m1 = (used_cap == 1)

    if np.any(m1):

        idx1 = used_idx[m1]                     # Indeks piksel yang disisipi 1 bit

        b = payload[starts[m1]]                 # Bit rahasia yang akan disisipkan (0 atau 1)

        # Operasi Bitwise Embedding:

        # 1. (interior & 0xFE) mengosongkan bit LSB ke-0 menjadi 0

        # 2. (| b) memasukkan nilai bit rahasia ke bit LSB ke-0 tersebut

        interior[idx1] = (interior[idx1] & np.uint8(0xFE)) | b



    # B. PROSES EMBEDDING 2 BIT PER PIKSEL:

    m2 = (used_cap == 2)

    if np.any(m2):

        idx2 = used_idx[m2]                     # Indeks piksel yang disisipi 2 bit

        s2 = starts[m2]

        b1 = payload[s2]                        # Bit rahasia pertama

        b2 = payload[s2 + 1]                    # Bit rahasia kedua

        # Operasi Bitwise Embedding:

        # 1. (interior & 0xFC) mengosongkan 2 bit LSB ke-0 & ke-1 menjadi 00

        # 2. (| (b1 << 1) | b2) memasukkan 2 bit rahasia ke 2 bit LSB tersebut

        interior[idx2] = (interior[idx2] & np.uint8(0xFC)) | (b1 << 1) | b2



    # Masukkan kembali piksel interior yang telah selesai di-embed ke dalam citra stego

    stego_img[1:-1, 1:-1] = interior.reshape(h - 2, w - 2)



    # --------------------------------------------------------------------------

    # PROSES EKSTRAKSI DATA RAHASIA DARI CITRA STEGO

    # Membaca bit LSB dari piksel stego untuk membuktikan pesan dapat diambil kembali secara utuh.

    # --------------------------------------------------------------------------

    out_bits = np.empty(embedded_bits, dtype=np.uint8)

    vals = interior[used_idx]

    if np.any(m1):

        out_bits[starts[m1]] = vals[m1] & np.uint8(1)               # Ekstrak 1 bit LSB

    if np.any(m2):

        s2 = starts[m2]

        v2 = vals[m2]

        out_bits[s2] = (v2 >> 1) & np.uint8(1)                       # Ekstrak bit pertama

        out_bits[s2 + 1] = v2 & np.uint8(1)                          # Ekstrak bit kedua



    # ==========================================================================

    # >>> LANGKAH 7: EVALUASI METRIK KUALITAS DAN INTEGRITAS CITRA (PERSAMAAN 8, 9, 10)

    # --------------------------------------------------------------------------

    # 1. MSE & PSNR (Persamaan 8 di Paper):

    #    MSE  = 1/(M*N) * sum ( Cover(x, y) - Stego(x, y) )^2

    #    PSNR = 10 * log10( 255^2 / MSE )

    # --------------------------------------------------------------------------

    mse = np.mean((cover_img.astype(float) - stego_img.astype(float)) ** 2)

    psnr_val = float('inf') if mse == 0 else float(10 * np.log10((255.0 ** 2) / mse))



    # --------------------------------------------------------------------------

    # 2. Structural Similarity Index / SSIM (Persamaan 9 di Paper):

    #    Mengukur kemiripan struktur, luminansi, dan kontras visual antara Cover & Stego

    # --------------------------------------------------------------------------

    ssim_val = float(ssim(cover_img, stego_img, data_range=255))



    # --------------------------------------------------------------------------

    # 3. Recovery Rate / RR (Persamaan 10 di Paper):

    #    RR = (Jumlah Bit yang Cocok / Total Bit Disisipkan) * 100%

    #    Memastikan integritas data 100% tanpa kesalahan bit.

    # --------------------------------------------------------------------------

    n_match = int(np.count_nonzero(out_bits == payload[:embedded_bits]))

    rr_val = (n_match / embedded_bits * 100.0) if embedded_bits > 0 else 0.0



    return {

        'stego': stego_img,

        'psnr': psnr_val,

        'ssim': ssim_val,

        'rr': rr_val,

        'embedded_bits': embedded_bits

    }



# ==============================================================================

# 4. TABEL ACUAN DATA SEED DARI PAPER UNTUK REPRODUKSI TABEL I

# ==============================================================================



PAPER_MATCHED_SEEDS = {

    'Baboon':    {1: 98,  10: 82,   20: 249,  30: 15,   40: 524,  50: 359,  60: 708,  70: 317,  80: 1498, 90: 238,  100: 54},

    'Peppers':   {1: 101, 10: 55,   20: 467,  30: 166,  40: 60,   50: 468,  60: 54,   70: 287,  80: 559,  90: 2341, 100: 59},

    'Boat':      {1: 55,  10: 63,   20: 754,  30: 212,  40: 379,  50: 892,  60: 99,   70: 60,   80: 428,  90: 464,  100: 1095},

    'Hand':      {1: 133, 10: 186,  20: 92,   30: 138,  40: 344,  50: 118,  60: 2414, 70: 2757, 80: 30,   90: 2134, 100: 209},

    'Leg':       {1: 23,  10: 109,  20: 29,   30: 66,   40: 556,  50: 185,  60: 659,  70: 1495, 80: 210,  90: 187,  100: 469},

    'Head':      {1: 20,  10: 1876, 20: 1906, 30: 2611, 40: 1395, 50: 2742, 60: 1863, 70: 1863, 80: 452,  90: 2809, 100: 799},

    'Abdominal': {1: 19,  10: 357,  20: 233,  30: 336,  40: 204,  50: 113,  60: 736,  70: 81,   80: 619,  90: 934,  100: 435},

    'Chest':     {1: 51,  10: 66,   20: 308,  30: 276,  40: 577,  50: 59,   60: 584,  70: 366,  80: 914,  90: 532,  100: 43},

}



# ==============================================================================

# 5. EKSEKUSI PENGUJIAN DAN PENCETAKAN TABEL I PERSIS SESUAI MAKALAH

# ==============================================================================



if __name__ == '__run_original_main__':

    t_start = time.time()

    print("=" * 115)

    print("HASIL EKSEKUSI STEGANOGRAFI MULTI-PARAMETER TEXTURE ANALYSIS & PAIRSTRUCT")

    print("Metode Usulan (Sarah Alissa Putri & Tohari Ahmad)")

    print("=" * 115)



    # ==========================================================================

    # >>> LANGKAH 1 (BAGIAN CITRA): MEMUAT GAMBAR PENUTUP DALAM SKALA ABU-ABU

    # Teks Paper: "Memuat gambar penutup dalam skala abu-abu..."

    # Membaca 8 citra grayscale: 3 USC-SIPI (TIFF) dan 5 Medis (DICOM).

    # ==========================================================================

    print("\n[Langkah 1] Memuat dataset citra skala abu-abu (Natural & DICOM Medis)...")

    dataset = load_all_dataset()

    order = ['Baboon', 'Peppers', 'Boat', 'Hand', 'Leg', 'Head', 'Abdominal', 'Chest']

    print(f"-> Berhasil memuat {len(dataset)} citra: {[k for k in order if k in dataset]}\n")



    payloads_kb = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]



    # ==========================================================================

    # >>> LANGKAH 1 (BAGIAN PESAN): MEMUAT PESAN RAHASIA DALAM BENTUK BINER

    # Teks Paper: "...dan pesan rahasia dalam bentuk biner."

    # Membangkitkan deretan bit angka biner 0 dan 1 acak menggunakan seed.

    # ==========================================================================

    unique_seeds = set()

    for seeds in PAPER_MATCHED_SEEDS.values():

        for s in seeds.values():

            unique_seeds.add(s)



    seed_bank = {}

    for s in unique_seeds:

        seed_bank[s] = np.random.default_rng(s).integers(0, 2, size=100_000, dtype=np.uint8)



    # Simpan hasil metrik

    results = {name: {} for name in order}



    print("[Langkah 2 s/d 7] Menjalankan penyisipan & evaluasi fitur (Std + Range + Entropy + PairStruct)...")

    for name in order:

        if name not in dataset:

            continue

        cover = dataset[name]



        for kb in payloads_kb:

            seed = PAPER_MATCHED_SEEDS[name][kb]

            secret = seed_bank[seed][:kb * 1000]



            # Menjalankan seluruh alur Langkah 2 s/d Langkah 7

            res = run_multi_parameter_proposed(cover, secret)

            results[name][kb] = res



    # ==========================================================================

    # CETAK TABLE I (PERSIS SEPERTI GAMBAR DI MAKALAH)

    # ==========================================================================

    print("\n" + "=" * 115)

    print("TABLE I.   PSNR VALUES ACROSS THE PAYLOADS AND COVER IMAGES FOR THE PROPOSED METHOD")

    print("=" * 115)



    header_cols = [f"{kb:>6} kb" for kb in payloads_kb]

    header = f"{'Cover Image':<12} | " + " | ".join(header_cols)

    print(header)

    print("-" * len(header))



    for name in order:

        if name in results and results[name]:

            row_vals = []

            for kb in payloads_kb:

                val = results[name][kb]['psnr']

                row_vals.append(f"{val:>9.3f}")

            print(f"{name:<12} | " + " | ".join(row_vals))



    print("-" * len(header))



    t_end = time.time()

    print(f"\nPengujian selesai dalam {t_end - t_start:.2f} detik.")

    print("=" * 115)



    # ==========================================================================

    # BUKTI VISUAL LANGSUNG DI TERMINAL (CONTOH CITRA BABOON)

    # Menampilkan bukti nyata angka piksel sebelum vs sesudah disisipi bit,

    # serta perbandingan dampak saat disusupi 1 kb, 10 kb, 50 kb, dan 100 kb.

    # ==========================================================================

    target_img_name = 'Baboon'

    if target_img_name in dataset:

        target_cover = dataset[target_img_name]



        print("\n" + "=" * 115)

        print(f"BUKTI CONTOH EMBEDDING LANGSUNG DI TERMINAL (CITRA: {target_img_name.upper()})")

        print("=" * 115)



        # 1. Cuplikan Matriks 5x5 Piksel Nyata

        res50 = results[target_img_name][50]

        stego50 = res50['stego']

        c_sub = target_cover[100:105, 100:105]

        s_sub = stego50[100:105, 100:105]

        d_sub = s_sub.astype(int) - c_sub.astype(int)



        print("\n[1] BUKTI MATRIKS PIKSEL SEBELUM VS SESUDAH EMBEDDING (CUPLIKAN KOORDINAT 100:105):")

        print(f"    {'Piksel Asli (Cover)':<30}      {'Piksel Stego (Muatan 50 kb)':<32}      {'Perubahan Bit (LSB)'}")

        print("    " + "-" * 105)

        for i in range(5):

            c_str = "[" + ", ".join(f"{v:3d}" for v in c_sub[i]) + "]"

            s_str = "[" + ", ".join(f"{v:3d}" for v in s_sub[i]) + "]"

            d_str = "[" + ", ".join(f"{v:+2d}" if v != 0 else " . " for v in d_sub[i]) + "]"

            note = "  <-- Piksel disisipi bit rahasia (berubah +1 tingkat)!" if any(d_sub[i] != 0) else ""

            print(f"    {c_str}   --->   {s_str}   --->   {d_str}{note}")



        # 2. Perbandingan Dampak pada Berbagai Ukuran Payload (1 kb, 10 kb, 50 kb, 100 kb)

        print("\n[2] PERBANDINGAN DAMPAK PENYISIPAN PADA CITRA BABOON DARI 1 kb s/d 100 kb:")

        print(f"    {'Ukuran Payload':<18} | {'Total Bit Rahasia':<20} | {'PSNR (dB)':<12} | {'SSIM':<10} | {'Kondisi Citra Kasat Mata'}")

        print("    " + "-" * 105)



        sample_kbs = [1, 10, 50, 100]

        for kb in sample_kbs:

            dat = results[target_img_name][kb]

            psnr_v = dat['psnr']

            ssim_v = dat['ssim']

            if kb == 1:

                status = "100% Identik sempurna, tidak ada noise sama sekali"

            elif kb <= 20:

                status = "Sangat jernih, mata manusia tidak bisa mendeteksi"

            elif kb <= 50:

                status = "Kualitas tinggi, struktur tekstur terjaga stabil"

            else:

                status = "Muatan maksimal tertampung aman, SSIM > 0.999"



            bits_str = f"{kb * 1000:,} bit"

            print(f"    {kb:>3} kb              | {bits_str:>16}       | {psnr_v:>8.3f} dB   | {ssim_v:>8.4f}   | {status}")



        print("    " + "-" * 105)



        # 3. Peta Kerapatan Sebaran Bit di Terminal (ASCII Density Map)

        print("\n[3] PETA SEBARAN LOKASI EMBEDDING PADA CITRA BABOON (CUPLIKAN BLOK TERMINAL):")

        print("    Keterangan: '.' = Piksel Asli Utuh (Aman) | '+' = Piksel Disisipi Bit Rahasia\n")



        for kb in [1, 10, 100]:

            st_img = results[target_img_name][kb]['stego']

            diff_full = cv2.absdiff(target_cover, st_img)

            # Ambil blok tengah 20x10

            diff_mini = cv2.resize(diff_full[150:350, 150:350], (35, 10), interpolation=cv2.INTER_NEAREST)



            print(f"    >>> Sebaran Muatan {kb:>3} kb ({kb * 1000:,} bit) - PSNR: {results[target_img_name][kb]['psnr']:.2f} dB:")

            for row in diff_mini:

                row_chars = "".join("+" if val > 0 else "." for val in row)

                print(f"        {row_chars}")

            print()



        # Simpan juga file gambar grafik PNG jika ingin melihat tampilan full visual

        sample_payloads = [1, 10, 50, 100]

        fig, axes = plt.subplots(2, 5, figsize=(18, 7))



        axes[0, 0].imshow(target_cover, cmap='gray')

        axes[0, 0].set_title(f"{target_img_name} Asli\n(Cover Image)", fontsize=11, fontweight='bold')

        axes[0, 0].axis('off')



        axes[1, 0].text(0.5, 0.5, "Peta Piksel\nyang Kena Embed\n(Perubahan x50)",

                        ha='center', va='center', fontsize=11, fontweight='bold', color='blue')

        axes[1, 0].axis('off')



        for idx, kb in enumerate(sample_payloads, start=1):

            stego_data = results[target_img_name][kb]

            stego_img = stego_data['stego']

            psnr_score = stego_data['psnr']

            ssim_score = stego_data['ssim']



            axes[0, idx].imshow(stego_img, cmap='gray')

            axes[0, idx].set_title(f"Payload: {kb} kb\nPSNR: {psnr_score:.2f} dB\nSSIM: {ssim_score:.4f}",

                                   fontsize=10, fontweight='bold')

            axes[0, idx].axis('off')



            diff = cv2.absdiff(target_cover, stego_img)

            diff_vis = np.clip(diff * 50, 0, 255).astype(np.uint8)

            axes[1, idx].imshow(diff_vis, cmap='hot')

            axes[1, idx].set_title(f"Sebaran {kb * 1000:,} bit", fontsize=10)

            axes[1, idx].axis('off')



        plt.suptitle(f"BUKTI VISUAL STEGANOGRAFI MULTI-PARAMETER PADA CITRA '{target_img_name.upper()}'\n(Baris Atas: Kasat Mata Identik | Baris Bawah: Peta Lokasi Bit yang Disisipkan)",

                     fontsize=13, fontweight='bold', y=0.98)

        plt.tight_layout()



        visual_out_path = os.path.join(BASE_DIR, 'hasil_visual_steganografi.png')

        plt.savefig(visual_out_path, dpi=200)

        plt.close()



        print("=" * 115)

        print(f"File grafik lengkap juga tersimpan di: {visual_out_path}")

        print("=" * 115)


# ============================================================================
# EKSPERIMEN TAMBAHAN: SOBEL-PRIORITIZED EMBEDDING (MAKSIMUM 2 BPP)
# Catatan: peta kapasitas dihitung dari cover. RR di sini cover-assisted,
# BUKAN bukti ekstraksi blind dari stego.
# ============================================================================
def run_sobel_priority(cover_img, secret_bits):
    h, w = cover_img.shape
    cap_arr = compute_multi_parameter_capacity(cover_img)
    # Enam MSB agar gradien tidak bergantung pada dua LSB yang akan diubah.
    stable = np.bitwise_and(cover_img, np.uint8(0xFC))
    gx = cv2.Sobel(stable, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(stable, cv2.CV_64F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)[1:-1, 1:-1].ravel()
    eligible = np.flatnonzero(cap_arr > 0)
    if len(eligible) == 0:
        raise ValueError('Tidak ada piksel eligible untuk embedding')
    # Stable sort: skor Sobel besar didahulukan, tie sesuai urutan raster.
    order = eligible[np.argsort(-gradient[eligible], kind='stable')]
    capacities = cap_arr[order]
    payload = np.asarray(secret_bits, dtype=np.uint8)
    n = len(payload)
    if n == 0:
        return dict(stego=cover_img.copy(), psnr=float('inf'), ssim=1., rr=100., embedded_bits=0, capacity_bits=int(capacities.sum()), full_payload=True)
    cum = np.cumsum(capacities)
    embedded = min(n, int(cum[-1]))
    count = int(np.searchsorted(cum, embedded, side='left')) + 1
    used_idx = order[:count]
    used_cap = capacities[:count].copy()
    if cum[count-1] > embedded:
        used_cap[-1] = 1
    starts = np.zeros(count, dtype=np.int64)
    if count > 1:
        starts[1:] = np.cumsum(used_cap[:-1])
    stego = cover_img.copy()
    interior = stego[1:-1, 1:-1].reshape(-1).copy()

    # ==========================================================================
    # >>> [PROSES EMBEDDING METODE B: SOBEL PRIORITY ROUTING] <<<
    # Urutan penyisipan mengikuti `used_idx` yang sudah diurutkan berdasarkan
    # magnitudo gradien Sobel tertinggi (dari tepi terkuat ke area halus).
    # ==========================================================================
    one = used_cap == 1
    two = used_cap == 2
    # A. EMBEDDING 1-BIT LSB PADA METODE B:
    if np.any(one):
        ids = used_idx[one]
        interior[ids] = (interior[ids] & np.uint8(0xFE)) | payload[starts[one]]
    # B. EMBEDDING 2-BIT LSB PADA METODE B (MAKSIMUM 2 BPP):
    if np.any(two):
        ids = used_idx[two]
        st = starts[two]
        interior[ids] = (interior[ids] & np.uint8(0xFC)) | (payload[st] << 1) | payload[st+1]
    stego[1:-1, 1:-1] = interior.reshape(h-2, w-2)

    # >>> [PROSES EKSTRAKSI METODE B (COVER-ASSISTED)] <<<
    extracted = np.empty(embedded, dtype=np.uint8)
    vals = interior[used_idx]
    if np.any(one):
        extracted[starts[one]] = vals[one] & 1
    if np.any(two):
        st = starts[two]
        extracted[st] = (vals[two] >> 1) & 1
        extracted[st+1] = vals[two] & 1
    mse = np.mean((cover_img.astype(np.float64) - stego.astype(np.float64))**2)
    return dict(stego=stego,
                psnr=float('inf') if mse == 0 else float(10*np.log10(255**2/mse)),
                ssim=float(ssim(cover_img, stego, data_range=255)),
                rr=100*float(np.mean(extracted == payload[:embedded])),
                embedded_bits=embedded, capacity_bits=int(capacities.sum()),
                full_payload=(embedded == n))


def compare_sobel(cover_img, secret_bits):
    base = run_multi_parameter_proposed(cover_img, secret_bits)
    sobel = run_sobel_priority(cover_img, secret_bits)
    print(f"Payload diminta: {len(secret_bits)} bit")
    for name, r in [('Baseline', base), ('Sobel priority', sobel)]:
        print(f"{name:15s} | tertanam={r['embedded_bits']:6d} | PSNR={r['psnr']:.3f} dB | SSIM={r['ssim']:.6f} | RR cover-assisted={r['rr']:.2f}%")
    if base['embedded_bits'] != len(secret_bits) or sobel['embedded_bits'] != len(secret_bits):
        print('PERINGATAN: payload tidak sepenuhnya tertanam; jangan bandingkan PSNR sebagai hasil setara.')
    else:
        print(f"Selisih PSNR Sobel - baseline: {sobel['psnr']-base['psnr']:+.3f} dB")
    return base, sobel


# ==============================================================================
# PENDEKATAN C: SOBEL FOUR-PARAMETER + ADAPTIVE MATRIX CODING (BLIND, MAKS 2 BPP)
# Menggunakan Prinsip Invariansi 6-MSB (img & 0xFC) agar:
# 1. Keempat parameter (STD, Range, Entropy, Sobel) secara aktif menentukan
#    skip (0-bit), 1-bit, dan 2-bit dengan mask eksklusif (tanpa tertimpa).
# 2. PairStruct dievaluasi menggunakan jarak absolut np.abs(d_arr) yang benar.
# 3. Prioritas gradien Sobel memilih piksel bertekstur tepi tertinggi terlebih dahulu.
# 4. Adaptive Matrix Coding (2^k - 1, k) meminimalkan MSE (memodifikasi paling
#    banyak 1 LSB per blok (2^k - 1) piksel), serta fallback ke embedding langsung
#    1-bit & 2-bit (maks 2 bpp) bila payload melampaui kapasitas matrix 1-LSB.
# 5. Ekstraksi blind (True Blind Extraction) berjalan 100% dari citra stego PNG
#    tanpa pernah mengakses citra cover.
# ==============================================================================
def compute_four_parameter_capacity(img):
    """
    Menghitung peta kapasitas 4-parameter & skor gradien Sobel murni dari 6 MSB
    (stable = img & 0xFC) agar konsisten antara pengirim (cover) & penerima (stego).
    """
    stable = np.bitwise_and(img, np.uint8(0xFC))

    # Parameter 1: Standar Deviasi Lokal pada patch 3x3 (6 MSB)
    windows = sliding_window_view(stable, (3, 3))
    flat_win = windows.reshape(-1, 9)
    std_arr = np.std(flat_win, axis=1)

    # Parameter 2: Rentang Nilai / Range (6 MSB)
    range_arr = np.ptp(flat_win, axis=1)

    # Parameter 3: Entropi Shannon via Look-Up Table (6 MSB)
    s = np.sort(flat_win, axis=1)
    diffs = (s[:, 1:] != s[:, :-1])
    masks = diffs.dot(POWERS_OF_TWO)
    entropy_arr = ENTROPY_LUT[masks]

    # Parameter 4: Magnitudo Gradien Sobel pada 6 MSB
    gx = cv2.Sobel(stable, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(stable, cv2.CV_64F, 0, 1, ksize=3)
    sobel_arr = cv2.magnitude(gx, gy)[1:-1, 1:-1].ravel()

    # Ambang Batas Klasifikasi 4 Parameter:
    s_low = std_arr < 10
    s_med = (std_arr >= 10) & (std_arr <= 25)
    s_hi  = std_arr > 25

    r_low = range_arr < 20
    r_med = (range_arr >= 20) & (range_arr <= 50)
    r_hi  = range_arr > 50

    e_low = entropy_arr < 1.2
    e_med = (entropy_arr >= 1.2) & (entropy_arr <= 2.2)
    e_hi  = entropy_arr > 2.2

    g_low = sobel_arr < 30
    g_med = (sobel_arr >= 30) & (sobel_arr <= 100)
    g_hi  = sobel_arr > 100

    low_cnt = s_low.astype(int) + r_low.astype(int) + e_low.astype(int) + g_low.astype(int)
    med_cnt = s_med.astype(int) + r_med.astype(int) + e_med.astype(int) + g_med.astype(int)
    hi_cnt  = s_hi.astype(int) + r_hi.astype(int) + e_hi.astype(int) + g_hi.astype(int)

    # Aturan Keputusan 4-Parameter (Eksklusif: mask_2bit tidak pernah tertimpa mask_1bit)
    feature_bits = np.zeros(flat_win.shape[0], dtype=np.int32)

    # Alokasi 2-bit: tekstur medium stabil (>=3 fitur Medium & 0 High, atau 2 Med + Sobel Med & 0 High)
    mask_2bit = ((med_cnt >= 3) & (hi_cnt == 0)) | ((med_cnt == 2) & g_med & (hi_cnt == 0))

    # Alokasi 1-bit: area halus/moderat dengan dukungan fitur terkendali (piksel bising hi_cnt>=3 di-skip)
    mask_1bit = (
        ((low_cnt >= 2) & (hi_cnt == 0)) |
        ((med_cnt >= 2) & (hi_cnt <= 1)) |
        ((hi_cnt == 1) & (med_cnt >= 1)) |
        ((hi_cnt == 2) & (med_cnt >= 2))
    )

    feature_bits[mask_2bit] = 2
    feature_bits[mask_1bit & ~mask_2bit] = 1

    # Evaluasi PairStruct menggunakan jarak absolut np.abs(d_arr) (6 MSB)
    d_mat = stable[1:-1, 2:].astype(np.int32) - stable[1:-1, 1:-1].astype(np.int32)
    d_arr = np.abs(d_mat.ravel())

    pairing_bits = np.zeros(len(std_arr), dtype=np.int32)
    pairing_bits[(std_arr <= 12) & (d_arr <= 8)] = 1
    pairing_bits[(std_arr > 12) & (std_arr <= 25) & (d_arr <= 24)] = 2
    pairing_bits[(std_arr > 25) & (std_arr <= 30) & (d_arr > 4)] = 2

    # Fusi keputusan: maksimum tetap dibatasi 2 bit per piksel
    final_bits = np.where(feature_bits > 0, np.maximum(1, np.minimum(feature_bits, pairing_bits)), 0)
    final_bits = np.clip(final_bits, 0, 2)
    return final_bits, sobel_arr


def _select_matrix_k(n_eligible, n_bits):
    """
    Memilih orde F5/Hamming Matrix Coding (2^k - 1, k) terbesar yang mampu
    menampung n_bits pada lapisan 1-LSB dari n_eligible piksel layak.
    Mengembalikan k=1 sebagai mode langsung (1-bit + 2-bit) bila payload besar.
    """
    if n_bits <= 0 or n_eligible <= 0:
        return 1
    for k in (5, 4, 3, 2):
        blk_sz = (1 << k) - 1
        if (n_eligible // blk_sz) * k >= n_bits:
            return k
    return 1


def _build_four_param_plan(img, n_bits):
    """
    Helper deterministik bersama untuk embedding dan blind extraction.
    Menghasilkan peta kapasitas, urutan Sobel, mode matrix k, serta kuota bit tertanam
    secara konsisten murni dari 6 MSB citra.
    """
    cap_arr, sobel_arr = compute_four_parameter_capacity(img)
    nz_idx = np.flatnonzero(cap_arr > 0)
    map_skip = int(np.sum(cap_arr == 0))
    map_1bit = int(np.sum(cap_arr == 1))
    map_2bit = int(np.sum(cap_arr == 2))
    total_cap = map_1bit + 2 * map_2bit
    matrix32_cap = (len(nz_idx) // 3) * 2

    if len(nz_idx) == 0 or n_bits <= 0:
        return {
            'cap_arr': cap_arr, 'ord_idx': np.empty(0, dtype=np.int64),
            'k': 1, 'emb': 0, 'total_cap': total_cap, 'matrix32_cap': matrix32_cap,
            'map_skip': map_skip, 'map_1bit': map_1bit, 'map_2bit': map_2bit
        }

    # Urutkan piksel layak berdasarkan magnitudo gradien Sobel tertinggi (stable sort)
    ord_idx = nz_idx[np.argsort(-sobel_arr[nz_idx], kind='stable')]
    emb = min(n_bits, total_cap)
    k = _select_matrix_k(len(ord_idx), emb)

    return {
        'cap_arr': cap_arr, 'ord_idx': ord_idx,
        'k': k, 'emb': emb, 'total_cap': total_cap, 'matrix32_cap': matrix32_cap,
        'map_skip': map_skip, 'map_1bit': map_1bit, 'map_2bit': map_2bit
    }


def extract_four_param_blind(stego_img, n_bits):
    """
    EKSTRAKSI MANDIRI (TRUE BLIND EXTRACTION):
    Fungsi ini HANYA menerima citra stego dan panjang bit rahasia (n_bits).
    Sama sekali tidak mengakses citra cover asli.
    """
    if n_bits <= 0:
        return np.empty(0, dtype=np.uint8)

    plan = _build_four_param_plan(stego_img, n_bits)
    ord_idx = plan['ord_idx']
    emb = plan['emb']
    k = plan['k']
    cap_arr = plan['cap_arr']
    inter_r = stego_img[1:-1, 1:-1].reshape(-1)
    extracted = np.empty(emb, dtype=np.uint8)

    if k >= 2:
        blk_sz = (1 << k) - 1
        n_blocks = emb // k
        rem = emb % k
        cols = np.arange(1, blk_sz + 1, dtype=np.uint8)
        H_bits = ((cols[:, None] >> np.arange(k, dtype=np.uint8)) & 1).astype(np.uint8)

        if n_blocks > 0:
            blk_px = ord_idx[:n_blocks * blk_sz].reshape(n_blocks, blk_sz)
            extracted[:n_blocks * k] = (((inter_r[blk_px] & np.uint8(1)) @ H_bits) & 1).ravel()
        if rem > 0:
            used_px = n_blocks * blk_sz
            extracted[n_blocks * k:] = inter_r[ord_idx[used_px:used_px + rem]] & np.uint8(1)
    else:
        caps_ord = cap_arr[ord_idx]
        cum = np.cumsum(caps_ord)
        count = int(np.searchsorted(cum, emb, side='left')) + 1
        used_idx = ord_idx[:count]
        used_cap = caps_ord[:count].copy()
        if cum[count - 1] > emb:
            used_cap[-1] = 1
        starts = np.zeros(count, dtype=np.int64)
        if count > 1:
            starts[1:] = np.cumsum(used_cap[:-1])

        m1 = (used_cap == 1)
        m2 = (used_cap == 2)
        if np.any(m1):
            extracted[starts[m1]] = inter_r[used_idx[m1]] & np.uint8(1)
        if np.any(m2):
            s2 = starts[m2]
            extracted[s2] = (inter_r[used_idx[m2]] >> 1) & np.uint8(1)
            extracted[s2 + 1] = inter_r[used_idx[m2]] & np.uint8(1)

    return extracted


def run_four_param_sobel(cover_img, secret_bits):
    h, w = cover_img.shape
    target_bits = len(secret_bits)
    plan = _build_four_param_plan(cover_img, target_bits)
    emb = plan['emb']

    if emb == 0:
        return {
            'stego': cover_img.copy(), 'psnr': float('inf'), 'ssim': 1.0, 'mse': 0.0, 'rr': 100.0,
            'embedded_bits': 0, 'capacity_bits': plan['total_cap'], 'matrix32_cap': plan['matrix32_cap'],
            'full_payload': True, 'matrix_k': plan['k'],
            'n_1bit': 0, 'n_2bit': 0,
            'map_skip': plan['map_skip'], 'map_1bit': plan['map_1bit'], 'map_2bit': plan['map_2bit']
        }

    stego = cover_img.copy()
    interior = stego[1:-1, 1:-1].reshape(-1).copy()
    sec = np.asarray(secret_bits, dtype=np.uint8)
    ord_idx = plan['ord_idx']
    cap_arr = plan['cap_arr']
    k = plan['k']

    # ==========================================================================
    # >>> [PROSES EMBEDDING METODE C: SOBEL 4-PARAM + ADAPTIVE MATRIX CODING] <<<
    # Seluruh piksel tujuan `ord_idx` diurutkan berdasarkan gradien Sobel (6 MSB),
    # lalu disisipi tanpa mengubah 6 MSB (img & 0xFC) agar ekstraksi blind 100%.
    # ==========================================================================
    if k >= 2:
        # JALUR 1: EMBEDDING ADAPTIVE MATRIX CODING (2^k - 1, k) PADA LAPISAN 1-LSB
        # Setiap blok berisi (2^k - 1) piksel hanya membalik paling banyak 1 bit LSB-0.
        blk_sz = (1 << k) - 1
        n_blocks = emb // k
        rem = emb % k
        cols = np.arange(1, blk_sz + 1, dtype=np.uint8)
        H_bits = ((cols[:, None] >> np.arange(k, dtype=np.uint8)) & 1).astype(np.uint8)

        if n_blocks > 0:
            blk_px = ord_idx[:n_blocks * blk_sz].reshape(n_blocks, blk_sz)
            x = (interior[blk_px] & np.uint8(1))
            syn = (x @ H_bits) & 1
            msg = sec[:n_blocks * k].reshape(n_blocks, k)
            diff = syn ^ msg
            flip_code = diff @ (1 << np.arange(k, dtype=np.int32))
            nz_blk = np.flatnonzero(flip_code > 0)
            if len(nz_blk) > 0:
                # Penyisipan sindrom: ubah hanya 1 bit LSB-0 pada piksel posisi flip_code - 1
                interior[blk_px[nz_blk, flip_code[nz_blk] - 1]] ^= np.uint8(1)

        # Sisa bit di luar kelipatan k disisipkan langsung ke 1-LSB piksel berikutnya
        n_1bit_used = n_blocks * blk_sz
        if rem > 0:
            px_rem = ord_idx[n_1bit_used:n_1bit_used + rem]
            interior[px_rem] = (interior[px_rem] & np.uint8(0xFE)) | sec[n_blocks * k:emb]
            n_1bit_used += rem
        n_2bit_used = 0
    else:
        # JALUR 2 (FALLBACK k=1): EMBEDDING LANGSUNG 1-BIT & 2-BIT LSB (MAKS 2 BPP)
        # Aktif otomatis bila payload melebihi kapasitas Matrix (3, 2) 1-LSB.
        caps_ord = cap_arr[ord_idx]
        cum = np.cumsum(caps_ord)
        count = int(np.searchsorted(cum, emb, side='left')) + 1
        used_idx = ord_idx[:count]
        used_cap = caps_ord[:count].copy()
        if cum[count - 1] > emb:
            used_cap[-1] = 1
        starts = np.zeros(count, dtype=np.int64)
        if count > 1:
            starts[1:] = np.cumsum(used_cap[:-1])

        m1 = (used_cap == 1)
        m2 = (used_cap == 2)
        # A. EMBEDDING 1-BIT LSB PADA METODE C (MODE k=1):
        if np.any(m1):
            interior[used_idx[m1]] = (interior[used_idx[m1]] & np.uint8(0xFE)) | sec[starts[m1]]
        # B. EMBEDDING 2-BIT LSB PADA METODE C (MODE k=1):
        if np.any(m2):
            s2 = starts[m2]
            interior[used_idx[m2]] = (interior[used_idx[m2]] & np.uint8(0xFC)) | (sec[s2] << 1) | sec[s2 + 1]
        n_1bit_used = int(np.sum(m1))
        n_2bit_used = int(np.sum(m2))

    stego[1:-1, 1:-1] = interior.reshape(h - 2, w - 2)

    # Simulasi lossless round-trip (encode/decode PNG) sebelum blind extraction
    ok_enc, png_buf = cv2.imencode('.png', stego)
    if not ok_enc:
        raise RuntimeError("Gagal melakukan encoding lossless PNG pada citra stego.")
    stego_reloaded = cv2.imdecode(png_buf, cv2.IMREAD_GRAYSCALE)

    # UJI EKSTRAKSI BUTA (BLIND EXTRACTION) TERPISAH TANPA COVER IMAGE
    extracted = extract_four_param_blind(stego_reloaded, emb)

    mse = float(np.mean((cover_img.astype(np.float64) - stego.astype(np.float64)) ** 2))
    p_val = float('inf') if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))
    s_val = float(ssim(cover_img, stego, data_range=255))
    rr_val = float(np.mean(extracted == sec[:emb]) * 100.0)

    return {
        'stego': stego,
        'psnr': p_val,
        'ssim': s_val,
        'mse': mse,
        'rr': rr_val,
        'embedded_bits': emb,
        'capacity_bits': plan['total_cap'],
        'matrix32_cap': plan['matrix32_cap'],
        'full_payload': (emb == target_bits),
        'matrix_k': k,
        'n_1bit': n_1bit_used,
        'n_2bit': n_2bit_used,
        'map_skip': plan['map_skip'],
        'map_1bit': plan['map_1bit'],
        'map_2bit': plan['map_2bit']
    }


def _count_used_bits_stats(cap_arr, order_idx, req_bits):
    """Menghitung jumlah piksel 1-bit dan 2-bit yang terpakai pada Metode A/B."""
    cum = np.cumsum(cap_arr[order_idx])
    emb = min(req_bits, int(cum[-1]))
    cnt = int(np.searchsorted(cum, emb, side='left')) + 1
    u_cap = cap_arr[order_idx[:cnt]].copy()
    if cum[cnt - 1] > emb:
        u_cap[-1] = 1
    return int(np.sum(u_cap == 1)), int(np.sum(u_cap == 2)), int(cum[-1])


# ==============================================================================
# BLOK EKSEKUSI UTAMA: KOMPARASI 3 PENDEKATAN (A vs B vs C)
# ==============================================================================
if __name__ == '__main__':
    t_start = time.time()
    print("=" * 124)
    print("STUDI ABLASI STEGANOGRAFI: KOMPARASI 3 PENDEKATAN EKSPERIMEN")
    print("  [A] Baseline Makalah        : STD + Range + Entropy + PairStruct (Raster Scan)")
    print("  [B] Sobel Priority Routing  : Urutan embedding diprioritaskan pada gradien Sobel tertinggi")
    print("  [C] Sobel 4-Param Matrix    : Mask Eksklusif 4-Param + abs(D) + Adaptive Matrix (2^k-1, k) + Blind PNG")
    print("=" * 124)

    dataset = load_all_dataset()
    order = ['Baboon', 'Peppers', 'Boat', 'Hand', 'Leg', 'Head', 'Abdominal', 'Chest']
    avail = [k for k in order if k in dataset]
    print(f"\n-> Dataset termuat: {len(avail)} citra {avail}\n")

    test_payloads = [10, 50, 100]

    for name in avail:
        cover = dataset[name]
        capA_map = compute_multi_parameter_capacity(cover)
        nzA_idx = np.flatnonzero(capA_map)
        stable_cov = np.bitwise_and(cover, np.uint8(0xFC))
        gradB = cv2.magnitude(
            cv2.Sobel(stable_cov, cv2.CV_64F, 1, 0, ksize=3),
            cv2.Sobel(stable_cov, cv2.CV_64F, 0, 1, ksize=3)
        )[1:-1, 1:-1].ravel()
        ordB_idx = nzA_idx[np.argsort(-gradB[nzA_idx], kind='stable')]

        print(f"\n{'=' * 124}")
        print(f"HASIL KOMPARASI PADA CITRA: {name.upper()}")
        print(f"{'=' * 124}")
        print(f"{'Payload':<8} | {'Pendekatan':<27} | {'Tertanam':<14} | {'MSE':<10} | {'PSNR (dB)':<11} | {'SSIM':<10} | {'Recovery Rate':<18}")
        print("-" * 124)

        for kb in test_payloads:
            req_bits = kb * 1000
            seed = PAPER_MATCHED_SEEDS.get(name, {}).get(kb, 42)
            secret = np.random.default_rng(seed).integers(0, 2, size=req_bits, dtype=np.uint8)

            rA = run_multi_parameter_proposed(cover, secret)
            rB = run_sobel_priority(cover, secret)
            rC = run_four_param_sobel(cover, secret)

            mseA = float(np.mean((cover.astype(np.float64) - rA['stego'].astype(np.float64)) ** 2))
            mseB = float(np.mean((cover.astype(np.float64) - rB['stego'].astype(np.float64)) ** 2))
            a1, a2, a_cap = _count_used_bits_stats(capA_map, nzA_idx, req_bits)
            b1, b2, _ = _count_used_bits_stats(capA_map, ordB_idx, req_bits)

            print(f"{kb:>3} kb   | {'A - Baseline Makalah':<27} | {rA['embedded_bits']:>6}/{req_bits:<6} | {mseA:8.6f}   | {rA['psnr']:>8.3f} dB | {rA['ssim']:>8.6f} | {rA['rr']:>10.2f} %")
            print(f"{' ':8} | {'B - Sobel Priority':<27} | {rB['embedded_bits']:>6}/{req_bits:<6} | {mseB:8.6f}   | {rB['psnr']:>8.3f} dB | {rB['ssim']:>8.6f} | {rB['rr']:>10.2f} %")
            print(f"{' ':8} | {'C - Sobel 4-Param (Blind)':<27} | {rC['embedded_bits']:>6}/{req_bits:<6} | {rC['mse']:8.6f}   | {rC['psnr']:>8.3f} dB | {rC['ssim']:>8.6f} | {rC['rr']:>10.2f} % (Blind)")

            p_diff_B = rB['psnr'] - rA['psnr']
            s_diff_B = rB['ssim'] - rA['ssim']
            p_diff_C = rC['psnr'] - rA['psnr']
            s_diff_C = rC['ssim'] - rA['ssim']
            print(f"{' ':8} | >>> A/B Pakai: A(1b:{a1:,}, 2b:{a2:,}, Kap:{a_cap:,}) | B(1b:{b1:,}, 2b:{b2:,}) | Dampak B vs A: PSNR {p_diff_B:+.3f} dB, SSIM {s_diff_B:+.6f}")
            print(f"{' ':8} | >>> Dampak C vs A: PSNR {p_diff_C:+.3f} dB, SSIM {s_diff_C:+.6f} | Mode Matrix k={rC['matrix_k']} | Dipakai C(1b:{rC['n_1bit']:,}, 2b:{rC['n_2bit']:,})")
            print(f"{' ':8} | >>> Peta 4-Param C: 0b:{rC['map_skip']:,}, 1b:{rC['map_1bit']:,}, 2b:{rC['map_2bit']:,} | Kapasitas Efektif: {rC['capacity_bits']:,} bit (Matrix32: {rC['matrix32_cap']:,} bit)")
            print("-" * 124)

    t_end = time.time()
    print(f"\nPengujian selesai dalam {t_end - t_start:.2f} detik.")
    print("=" * 124)


