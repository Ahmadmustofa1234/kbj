# Laporan Implementasi dan Verifikasi Metode Steganografi

**Makalah acuan:** *Enhanced Image Steganography Through Multi-Parameter Texture Analysis*, Sarah Alissa Putri & Tohari Ahmad (ICICyTA 2025)
**File implementasi:** `test1_steganography.py` (evaluasi lengkap) dan `training.py` (uji cepat 20 kb)

---

## 1. Tujuan

Laporan ini menjelaskan implementasi ulang metode pada makalah acuan, memeriksa kesesuaian setiap rumus dengan kode, lalu membandingkan hasil eksperimen dengan Tabel I, II, dan III pada makalah.

## 2. Kesesuaian Rumus dan Metodologi

| No | Komponen di Makalah | Penjelasan | Implementasi |
|---|---|---|---|
| 1 | Pers. (1): Patch 3×3 | Setiap piksel diambil bersama 8 tetangganya. Piksel tepi tidak diproses. | `sliding_window_view(img, (3, 3))` |
| 2 | Pers. (2): Standar deviasi (σ) | Variasi intensitas di dalam patch, dengan pembagi N = 9. | `np.std(..., axis=1)` |
| 3 | Pers. (3): Range (R) | Selisih nilai maksimum dan minimum di dalam patch. | `np.ptp(..., axis=1)` |
| 4 | Pers. (4): Entropi Shannon (H) | Kerumitan tekstur, −Σ pᵢ log₂ pᵢ. | Lookup table entropi untuk 9 piksel |
| 5 | Threshold dan Case A–D | σ (10, 25), R (20, 50), H (1,2, 2,2). Case A: 1 bit. Case B: 2 bit. Case C: 1 bit. Case D: dilewati. | `compute_pixel_capacity_map` |
| 6 | Pers. (5) dan Fig. 1: PairStruct | D(i, j) = I(i, j+1) − I(i, j). σ ≤ 15 dan D ≤ 4: 1 bit. 15 < σ ≤ 20 dan D ≤ 4: 2 bit. 20 < σ ≤ 30 dan D > 4: 2 bit. Selain itu: 0 bit. | `compute_pixel_capacity_map` |
| 7 | Pers. (6): Kapasitas akhir | final = max(1, min(feature_bits, pairing_bits)) untuk piksel yang tidak dilewati. | `np.where(feature_bits > 0, ...)` |
| 8 | Penyisipan dan ekstraksi | Bit disisipkan ke 1 atau 2 LSB. Peta kapasitas dihitung ulang dari cover saat ekstraksi. | `embed_data`, `extract_data` |
| 9 | Pers. (7)–(10) | MSE, PSNR, SSIM, dan Recovery Rate (RR). | `calculate_psnr`, `calculate_ssim` |

## 3. Asumsi Implementasi

Beberapa detail tidak disebutkan secara eksplisit di makalah, sehingga implementasi ini memakai asumsi berikut:

1. **Pers. (6) vs Fig. 1.** Pers. (6) menulis max(1, min(...)), sedangkan flowchart Fig. 1 menulis min(...). Implementasi ini memakai Pers. (6) karena hasilnya lebih dekat ke Tabel I.
2. **Satuan payload.** 1 kb diasumsikan 1000 bit. Makalah tidak mendefinisikan satuan ini.
3. **Preprocessing.** Makalah menyebut bilateral denoising dan Gaussian filtering, tetapi tidak mencantumkan parameternya. Pengujian ini dilakukan **tanpa preprocessing**.
4. **Payload acak.** Makalah memakai *random binary data* tanpa menyebut seed. Pengujian ini memakai `np.random.default_rng(42)`.
5. **Citra medis.** Makalah tidak menyebut file DICOM yang dipakai. Lima citra DICOM dari DICOM Library dipakai sebagai pengganti, lalu dikonversi ke grayscale dan diubah ukurannya menjadi 512×512.
6. **Urutan bit.** Untuk slot 2 bit, bit pertama payload disimpan di bit-1 dan bit kedua di bit-0.

## 4. Dataset

| Citra | Kategori | Kapasitas maksimum |
|---|---|---|
| Baboon | Alami (USC-SIPI) | 139,7 kb |
| Peppers | Alami (USC-SIPI) | 215,0 kb |
| Boat | Alami (USC-SIPI) | 206,2 kb |
| Hand | Medis (DICOM) | 169,8 kb |
| Leg | Medis (DICOM) | 216,6 kb |
| Head | Medis (DICOM) | 184,4 kb |
| Abdominal | Medis (DICOM) | 181,3 kb |
| Chest | Medis (DICOM) | 227,1 kb |

Semua citra berukuran 512×512 grayscale. Kapasitas semua citra melebihi payload terbesar yang diuji (100 kb).

## 5. Hasil Eksperimen (seed = 42, 1 kb = 1000 bit)

### Tabel 1. PSNR (dB) hasil implementasi

| Citra | 1 kb | 10 kb | 20 kb | 30 kb | 40 kb | 50 kb | 60 kb | 70 kb | 80 kb | 90 kb | 100 kb |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baboon | 75,008 | 65,277 | 62,298 | 60,549 | 59,315 | 58,348 | 57,554 | 56,877 | 56,296 | 55,777 | 55,317 |
| Peppers | 75,379 | 65,354 | 62,316 | 60,563 | 59,313 | 58,344 | 57,545 | 56,878 | 56,296 | 55,789 | 55,331 |
| Boat | 75,327 | 65,321 | 62,296 | 60,561 | 59,303 | 58,331 | 57,535 | 56,866 | 56,292 | 55,783 | 55,324 |
| Hand | 75,318 | 65,329 | 62,340 | 60,546 | 59,297 | 58,326 | 57,538 | 56,876 | 56,293 | 55,779 | 55,316 |
| Leg | 75,468 | 65,396 | 62,357 | 60,587 | 59,332 | 58,368 | 57,563 | 56,880 | 56,300 | 55,791 | 55,328 |
| Head | 75,266 | 65,269 | 62,265 | 60,515 | 59,273 | 58,317 | 57,531 | 56,861 | 56,278 | 55,776 | 55,319 |
| Abdominal | 75,275 | 65,290 | 62,276 | 60,568 | 59,296 | 58,316 | 57,540 | 56,877 | 56,293 | 55,780 | 55,313 |
| Chest | 75,468 | 65,412 | 62,394 | 60,604 | 59,342 | 58,342 | 57,546 | 56,875 | 56,287 | 55,773 | 55,308 |

### Perbandingan dengan Tabel I makalah

| Rentang payload | Selisih maksimum terhadap makalah |
|---|---|
| 1 kb | hingga ±0,42 dB |
| 10 kb | hingga ±0,24 dB |
| 20–100 kb | ≤ 0,12 dB |

Pada payload 20–100 kb, hasil implementasi sangat dekat dengan makalah. Selisih terbesar ada pada payload kecil karena PSNR di sana sangat bergantung pada isi bit acak. Pada 1 kb hanya sekitar 500 piksel yang berubah, sehingga selisih beberapa puluh piksel saja sudah menggeser PSNR beberapa persepuluh dB.

**Uji sensitivitas seed.** Pengujian ulang dengan 30 seed berbeda (seed 0–29) pada Baboon, Peppers, dan Boat menunjukkan bahwa PSNR 1 kb bervariasi sekitar 75,0–75,65 dB. Nilai makalah berada di dalam rentang ini. Rata-rata dari 30 seed selisihnya < 0,06 dB dari makalah untuk hampir semua payload (data: `steganography_seed_sensitivity_B_equation6_1000bits_all_payloads_summary.csv`).

**Catatan Tabel I makalah.** Nilai Baboon 40 kb di makalah (60,552 dB) sama persis dengan nilai Baboon 30 kb. Kemungkinan ini salah ketik, karena PSNR seharusnya turun ketika payload bertambah. Hasil implementasi (59,315 dB) konsisten dengan tujuh citra lainnya di kolom 40 kb.

### Tabel 2. Baboon: makalah vs implementasi

| Payload | PSNR Prior [7] | PSNR makalah | PSNR implementasi | SSIM makalah | SSIM implementasi | RR implementasi |
|---|---|---|---|---|---|---|
| 40 kb | 59,162 | 60,552* | 59,315 | 1,000 | 1,000 | 100% |
| 50 kb | 57,626 | 58,362 | 58,348 | 1,000 | 1,000 | 100% |
| 60 kb | 56,856 | 57,558 | 57,554 | 0,999 | 1,000 | 100% |
| 70 kb | 56,158 | 56,872 | 56,877 | 0,999 | 1,000 | 100% |
| 80 kb | 55,629 | 56,266 | 56,296 | 0,999 | 0,999 | 100% |
| 90 kb | 55,208 | 55,780 | 55,777 | 0,999 | 0,999 | 100% |
| 100 kb | 54,832 | 55,320 | 55,317 | 0,999 | 0,999 | 100% |

\* Diduga salah ketik (lihat catatan Tabel I).

### Tabel 3. Head: makalah vs implementasi

| Payload | PSNR Prior [7] | PSNR makalah | PSNR implementasi | SSIM implementasi | RR implementasi |
|---|---|---|---|---|---|
| 40 kb | 58,626 | 59,195 | 59,273 | 1,000 | 100% |
| 50 kb | 57,656 | 58,272 | 58,317 | 0,999 | 100% |
| 60 kb | 56,854 | 57,468 | 57,531 | 0,999 | 100% |
| 70 kb | 56,251 | 56,812 | 56,861 | 0,999 | 100% |
| 80 kb | 55,749 | 56,263 | 56,278 | 0,999 | 100% |
| 90 kb | 55,205 | 55,755 | 55,776 | 0,999 | 100% |
| 100 kb | 54,715 | 55,298 | 55,319 | 0,999 | 100% |

Citra Head hasil implementasi tidak sama dengan citra Head di makalah, sehingga selisih kecil pada tabel ini wajar.

## 6. Kesimpulan

1. Semua rumus pada makalah (Pers. 1–10) beserta aturan Case A–D dan PairStruct sudah diimplementasikan.
2. Pada payload 20–100 kb, hasil PSNR selisihnya ≤ 0,12 dB dari Tabel I makalah. Selisih yang lebih besar pada 1–10 kb dijelaskan oleh sifat acak payload.
3. Recovery Rate 100% pada semua citra dan semua ukuran payload, dan SSIM ≥ 0,999.
4. Sesuai klaim makalah, PSNR metode yang diusulkan lebih tinggi daripada metode Prior [7] di semua ukuran payload pada Tabel 2 dan Tabel 3.
5. Selisih yang tersisa terhadap makalah berasal dari detail yang tidak dicantumkan di makalah: seed acak, parameter preprocessing, satuan kb, dan file DICOM yang dipakai.

## 7. Cara Menjalankan

```powershell
cd "G:\My Drive\kbj"
python test1_steganography.py   # Tabel 1, 2, dan 3
python training.py              # Uji cepat payload 20 kb
```
