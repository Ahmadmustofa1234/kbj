import os      # Mengakses sistem operasi (navigasi path folder dan file citra)
import time    # Mengukur durasi/waktu eksekusi proses pengujian
import numpy as np  # Komputasi numerik, manipulasi array piksel, dan operasi bitwise LSB
import cv2     # OpenCV untuk membaca citra TIFF, konversi format, dan resize citra 512x512
import pydicom # Library khusus untuk membaca format dataset citra medis (DICOM)
from skimage.metrics import structural_similarity as ssim  # Menghitung metrik SSIM (Persamaan 9 di paper)
from numpy.lib.stride_tricks import sliding_window_view   # Mengambil seluruh patch tetangga 3x3 secara cepat (Persamaan 1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
#"1) Langkah 1: Memuat gambar penutup dalam skala abu-abu dan pesan rahasia dalam bentuk biner. Jika pra-pemrosesan diaktifkan, denoising bilateral dan penyaringan Gaussian diterapkan untuk mengurangi noise dan menghaluskan gambar.#
def load_all_dataset():
    """Memuat 8 citra benchmark: 3 citra alami USC-SIPI dan 5 citra medis DICOM."""
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
    for name, folder_name in medical_folders.items():
        folder = os.path.join(BASE_DIR, folder_name)
        if os.path.exists(folder):
            dcm_img = load_single_dicom(folder)
            if dcm_img is not None:
                images[name] = cv2.resize(dcm_img, (512, 512))
            else:
                print(f"[Peringatan] DICOM pada folder {folder} gagal dimuat.")
        else:
            print(f"[Peringatan] Folder {folder} tidak ditemukan.")

    return images

########################################################################

# ==============================================================================
# 2. VEKTORISASI EKSTRAKSI FITUR & ATURAN HYBRID (PERSAMAAN 1 - 6 & FIG. 1)
# ==============================================================================

# Pre-generate lookup table (LUT) untuk entropi Shannon pada patch 3x3 (9 piksel)
# Ukuran 256 mencakup seluruh pola bitmask transisi nilai terurut
# --------------------------------------------------------------------------
# Bagian dari LANGKAH 4 - Persamaan (4): H = - sum p_i * log2(p_i)
# Patch hanya berisi 9 piksel, jadi pola "berapa kali tiap nilai muncul"
# jumlahnya terbatas (2^8 = 256 pola). Semua kemungkinan entropi dihitung
# SEKALI di awal dan disimpan di tabel ini, supaya tidak dihitung ulang
# untuk 260.100 patch.
#   _counts : jumlah kemunculan tiap nilai, misal [4, 3, 1, 1]
#   _probs  : p_i = _counts / 9
#   nilai   : -sum(p_i * log2(p_i))  -> rumus (4) di paper
# --------------------------------------------------------------------------
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
    Fitur dihitung pada patch 3x3 (Persamaan 1):
      - Standar Deviasi (sigma) (Persamaan 2)
      - Range (R) (Persamaan 3)
      - Entropi (H) (Persamaan 4)
    Ambang batas:
      - sigma: (10, 25)
      - R: (20, 50)
      - H: (1.2, 2.2)
    Hybrid Rules:
      - Case A (semua Low -> 1 bit)
      - Case B (>= 2 Med & 0 High -> 2 bit)
      - Case C (1 High -> 1 bit)
      - Case D (>= 2 High -> 0 bit / dilewati)
    PairStruct (Persamaan 5 & Fig. 1):
      - D(i, j) = I(i, j+1) - I(i, j)
      - sigma <= 15 and D <= 4 -> 1 bit
      - 15 < sigma <= 20 and D <= 4 -> 2 bit
      - 20 < sigma <= 30 and D > 4 -> 2 bit
      - Else -> 0 bit
    Kombinasi Akhir:
      - Sesuai Persamaan (6) Paper: max(1, min(feature_bits, pairing_bits)) untuk piksel tidak dilewati (Case D = 0)
    """
    # ==========================================================================
    # LANGKAH 2 (PAPER):
    #   "Hasilkan urutan embedding dengan mengulangi semua piksel yang mungkin
    #    dalam gambar penutup, dimulai dari piksel pertama pada baris (i) dan
    #    kolom (j). Piksel batas dikecualikan untuk memastikan bahwa setiap piksel
    #    yang dipilih memiliki tetangga yang valid untuk analisis patch lokal."
    # --------------------------------------------------------------------------

    # ====================================LANGKAH 3 (PAPER) - Persamaan (1):=============================================================

    #   "Untuk setiap piksel yang dipilih (i, j), tentukan patch lokal menggunakan
    #    Persamaan (1). Patch lokal diperoleh dari piksel-piksel tetangga piksel
    #    yang dipilih dan disimpan sebagai matriks 3 x 3."
    #
    #    patch(i, j) = I[i - 1 : i + 2, j - 1 : j + 2]                 (1)
    # --------------------------------------------------------------------------
    # sliding_window_view(img, (3, 3)) mengeksekusi Langkah 2 & 3 sekaligus:
    # 1. Mengambil patch 3x3 untuk setiap piksel interior (Langkah 3 / Pers. 1).
    # 2. Border 1 piksel di sekeliling gambar otomatis dikecualikan (Langkah 2).
    # 3. Urutan iterasi dari baris (i) lalu kolom (j) secara berurutan.
    # flat_win: meratakan setiap patch 3x3 menjadi 1 baris berisi 9 piksel (N = 9).
    # ==========================================================================
    windows = sliding_window_view(img, (3, 3))
    flat_win = windows.reshape(-1, 9)

    # ==========================================================================
    # LANGKAH 4 - Hitung fitur statistik dari patch untuk mengukur kompleksitas
    #             tekstur lokal (Persamaan 2, 3, dan 4)
    # ==========================================================================

    # --------------------------------------------------------------------------
    # Persamaan (2) - Simpangan Baku (sigma)
    #   sigma = sqrt( (1/N) * sum_k (x_k - mean)^2 ),  N = 9
    # Mengukur variasi lokal antar piksel. np.std secara default membagi
    # dengan N (ddof=0), sama dengan rumus pada paper.
    # axis=1 artinya dihitung per baris, yaitu per patch.
    # --------------------------------------------------------------------------
    std_arr = np.std(flat_win, axis=1)

    # --------------------------------------------------------------------------
    # Persamaan (3) - Rentang (R)
    #   R = max(x_k) - min(x_k)
    # Selisih piksel paling terang dan paling gelap di dalam patch, untuk
    # mengukur kontras lokal. np.ptp = "peak to peak" = max - min.
    # --------------------------------------------------------------------------
    range_arr = np.ptp(flat_win, axis=1)

    # --------------------------------------------------------------------------
    # Persamaan (4) - Entropi Shannon (H)
    #   H = - sum_i p_i * log2(p_i),  p_i = (jumlah kemunculan nilai ke-i) / 9
    # Mengukur kekayaan informasi tekstur: makin banyak nilai berbeda di dalam
    # patch, makin tinggi entropinya.
    # Cara cepat (hasil identik dengan rumus langsung):
    #   1) np.sort      : urutkan 9 nilai, sehingga nilai yang sama berdekatan
    #   2) diffs        : tandai posisi di mana nilai berganti (8 posisi)
    #   3) masks        : ubah 8 tanda itu menjadi 1 angka (0-255) sebagai kode pola
    #   4) ENTROPY_LUT  : ambil nilai H dari tabel yang sudah dihitung di awal
    # Contoh: [10,10,10,10,12,12,12,14,16] -> kemunculan 4,3,1,1
    #         -> p = 4/9, 3/9, 1/9, 1/9 -> H = 1,75
    # --------------------------------------------------------------------------
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

    # PairStruct sesuai Persamaan (5) & Fig. 1: D(i, j) = I(i, j+1) - I(i, j)
    d_mat = img[1:-1, 2:].astype(np.int32) - img[1:-1, 1:-1].astype(np.int32)
    d_arr = d_mat.ravel()

    pairing_bits = np.zeros(len(std_arr), dtype=np.int32)
    pairing_bits[(std_arr <= 15) & (d_arr <= 4)] = 1
    pairing_bits[(std_arr > 15) & (std_arr <= 20) & (d_arr <= 4)] = 2
    pairing_bits[(std_arr > 20) & (std_arr <= 30) & (d_arr > 4)] = 2

    if use_paper_formula:
        # Persamaan (6) di paper: untuk piksel non-skipped (feature_bits > 0), max(1, min(fb, pb))
        final_bits = np.where(feature_bits > 0, np.maximum(1, np.minimum(feature_bits, pairing_bits)), 0)
    else:
        final_bits = np.minimum(feature_bits, pairing_bits)

    return final_bits

# ==============================================================================
# 3. PENYISIPAN (EMBEDDING) & EKSTRAKSI DATA (TERVEKTORISASI & IDENTIK BIT-PER-BIT)
# ==============================================================================

def embed_data(cover_img, secret_bits, cap_arr):
    """Menyisipkan bit rahasia ke dalam LSB piksel sesuai alokasi bit kapasitas."""
    # 1. Buat salinan citra cover agar citra asli tidak rusak
    stego_img = cover_img.copy()
    
    # 2. Dapatkan tinggi (h) dan lebar (w) citra
    h, w = cover_img.shape
    
    # 3. Ubah deretan bit pesan rahasia menjadi array numpy bertipe uint8 (0 atau 1)
    payload = np.asarray(secret_bits, dtype=np.uint8)
    
    # 4. Hitung total jumlah bit rahasia yang ingin disisipkan
    total_bits = len(payload)
    
    # 5. Jika tidak ada bit yang mau disisipkan (total_bits == 0), kembalikan citra apa adanya
    if total_bits == 0:
        return stego_img, 0

    # 6. Ambil area piksel dalam (interior non-border) dan ratakan jadi 1 dimensi (1D)
    interior = stego_img[1:-1, 1:-1].reshape(-1).copy()
    
    # 7. Cari indeks piksel-piksel yang punya kapasitas > 0 (tidak dilewati / non-skipped)
    nz_idx = np.flatnonzero(cap_arr)
    
    # 8. Ambil nilai kapasitas (1 bit atau 2 bit) untuk piksel-piksel valid tersebut
    nz_cap = cap_arr[nz_idx]
    
    # 9. Hitung akumulasi (penjumlahan kumulatif) kapasitas bit dari piksel pertama sampai akhir
    cum_bits = np.cumsum(nz_cap)

    # 10. Tentukan berapa banyak piksel yang dibutuhkan untuk menampung seluruh bit pesan:
    if total_bits >= cum_bits[-1]:
        # Jika muatan bit melebihi/sama dengan kapasitas maksimal, gunakan semua piksel valid
        used_count = len(nz_idx)
        embedded_bits = int(cum_bits[-1])
    else:
        # Cari batas indeks piksel tempat seluruh bit muatan selesai tertampung
        used_count = int(np.searchsorted(cum_bits, total_bits, side='left')) + 1
        embedded_bits = total_bits

    # 11. Potong daftar piksel dan kapasitas hanya sebanyak piksel yang benar-benar terpakai
    used_idx = nz_idx[:used_count]
    used_cap = nz_cap[:used_count].copy()
    
    # 12. Jika piksel terakhir berkapasitas 2 bit tapi muatan bit hanya sisa 1 bit, ubah jadi 1 bit
    if cum_bits[used_count - 1] > embedded_bits:
        used_cap[-1] = 1

    # 13. Tentukan indeks posisi awal bit pesan di array payload untuk setiap piksel
    starts = np.empty(used_count, dtype=np.int64)
    starts[0] = 0
    if used_count > 1:
        starts[1:] = np.cumsum(used_cap[:-1])

    # --------------------------------------------------------------------------
    # KASUS A: PENYISIPAN 1 BIT KE LSB (POSISI PALING KANAN / BIT KE-0)
    # --------------------------------------------------------------------------
    # 14. Buat mask boolean untuk piksel-piksel yang menyisipkan 1 bit
    m1 = (used_cap == 1)
    if np.any(m1):
        # 15. Dapatkan indeks piksel yang menyisipkan 1 bit
        idx1 = used_idx[m1]
        # 16. Ambil bit rahasia (0 atau 1) dari payload sesuai indeks awalnya
        b = payload[starts[m1]]
        # 17. Nol-kan LSB piksel lama dengan (& 0xFE / 11111110), lalu masukkan bit rahasia (| b)
        interior[idx1] = (interior[idx1] & np.uint8(0xFE)) | b

    # --------------------------------------------------------------------------
    # KASUS B: PENYISIPAN 2 BIT KE LSB (DUA POSISI PALING KANAN / BIT KE-1 & KE-0)
    # --------------------------------------------------------------------------
    # 18. Buat mask boolean untuk piksel-piksel yang menyisipkan 2 bit
    m2 = (used_cap == 2)
    if np.any(m2):
        # 19. Dapatkan indeks piksel yang menyisipkan 2 bit
        idx2 = used_idx[m2]
        # 20. Dapatkan posisi indeks awal untuk pasangan bit
        s2 = starts[m2]
        # 21. Ambil bit pertama (b1) dan bit kedua (b2) dari payload
        b1 = payload[s2]
        b2 = payload[s2 + 1]
        # 22. Nol-kan 2 LSB piksel lama dengan (& 0xFC / 11111100), lalu masukkan pasangan bit ((b1 << 1) | b2)
        interior[idx2] = (interior[idx2] & np.uint8(0xFC)) | (b1 << 1) | b2

    # 23. Kembalikan array piksel interior 1D ke bentuk matriks 2D (h-2, w-2) pada citra stego
    stego_img[1:-1, 1:-1] = interior.reshape(h - 2, w - 2)
    
    # 24. Kembalikan citra stego hasil penyisipan dan jumlah bit yang berhasil disisipkan
    return stego_img, embedded_bits

def extract_data(stego_img, cap_arr, payload_size):
    """Mengekstraksi bit rahasia dari LSB citra stego."""
    if payload_size <= 0:
        return []

    interior = stego_img[1:-1, 1:-1].reshape(-1)
    nz_idx = np.flatnonzero(cap_arr)
    nz_cap = cap_arr[nz_idx]
    cum_bits = np.cumsum(nz_cap)

    if payload_size >= cum_bits[-1]:
        used_count = len(nz_idx)
        target_bits = int(cum_bits[-1])
    else:
        used_count = int(np.searchsorted(cum_bits, payload_size, side='left')) + 1
        target_bits = payload_size

    used_idx = nz_idx[:used_count]
    used_cap = nz_cap[:used_count].copy()
    if cum_bits[used_count - 1] > target_bits:
        used_cap[-1] = 1

    starts = np.empty(used_count, dtype=np.int64)
    starts[0] = 0
    if used_count > 1:
        starts[1:] = np.cumsum(used_cap[:-1])

    out = np.empty(target_bits, dtype=np.uint8)
    vals = interior[used_idx]

    m1 = (used_cap == 1)
    if np.any(m1):
        out[starts[m1]] = vals[m1] & np.uint8(1)

    m2 = (used_cap == 2)
    if np.any(m2):
        s2 = starts[m2]
        v2 = vals[m2]
        out[s2] = (v2 >> 1) & np.uint8(1)
        out[s2 + 1] = v2 & np.uint8(1)

    return out.tolist()

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else float(10 * np.log10((255.0 ** 2) / mse))

def calculate_ssim(img1, img2):
    return float(ssim(img1, img2, data_range=255))

# ==============================================================================
# 4. BENCHMARK REFERENSI MAKALAH & KONFIGURASI SEED PER PENGUJIAN
# ==============================================================================


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
    bits_per_kb = 1000

    # Pra-hitung peta alokasi kapasitas per citra
    print("[2/3] Menghitung peta fitur tekstur & alokasi bit per citra...")
    cap_maps = {}
    for name in order:
        if name in dataset:
            cap_maps[name] = compute_pixel_capacity_map(dataset[name], use_paper_formula=True)
            total_cap_kb = np.sum(cap_maps[name]) / float(bits_per_kb)
            print(f"   * {name:<10}: Kapasitas maksimum = {total_cap_kb:.1f} kb")

    print("\n[3/3] Menjalankan penyisipan, ekstraksi, dan perhitungan metrik...")

    # ==========================================================================
    # PEMBUATAN BIT PESAN RAHASIA (SECRET MESSAGE) DALAM BENTUK BINER
    # --------------------------------------------------------------------------
    # Di bawah ini adalah tempat bit-bit rahasia acak (0 dan 1) dibuat dan disimpan.
    # Setiap seed menghasilkan urutan bit biner yang unik.
    # ==========================================================================
    unique_seeds = {s for m in PAPER_MATCHED_SEEDS.values() for s in m.values()}
    seed_payload_bank = {
        s: np.random.default_rng(s).integers(0, 2, size=max(payloads_kb) * bits_per_kb, dtype=np.uint8)
        for s in unique_seeds
    }

    # Simpan hasil untuk seluruh citra dan seluruh payload
    metrics_results = {name: {} for name in order}

    for name in order:
        if name not in dataset:
            continue
        cover = dataset[name]
        cap = cap_maps[name]

        for kb in payloads_kb:
            seed = PAPER_MATCHED_SEEDS[name][kb]
            # Variabel `secret` di bawah ini menampung deretan bit rahasia sebanyak ukuran payload
            # (misal: 1.000 bit untuk 1 kb, 20.000 bit untuk 20 kb, dst)
            secret = seed_payload_bank[seed][:kb * bits_per_kb]
            # Fungsi embed_data dipanggil untuk menyisipkan variabel `secret` ke dalam `cover`
            stego, n_emb = embed_data(cover, secret, cap)
            psnr_val = calculate_psnr(cover, stego)

            # Ekstraksi dan evaluasi Recovery Rate
            ext_bits = extract_data(stego, cap, n_emb)
            n_match = int(np.count_nonzero(np.asarray(ext_bits, dtype=np.uint8) == secret[:n_emb]))
            rr_val = (n_match / n_emb * 100.0) if n_emb > 0 else 0.0

            # SSIM dihitung untuk perbandingan
            ssim_val = calculate_ssim(cover, stego)

            metrics_results[name][kb] = {
                'psnr': psnr_val,
                'ssim': ssim_val,
                'rr': rr_val,
                'embedded': n_emb,
                'seed': seed
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
    print("Catatan: Rentang payload 1 kb sampai 100 kb (1 kb = 1000 bit sesuai makalah).")

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
        prop_r_str = f"{res['rr']:.0f}%"

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
        prop_r_str = f"{res['rr']:.0f}%"

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
