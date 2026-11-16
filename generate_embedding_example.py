import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
from fix import load_all_dataset, run_multi_parameter_proposed

ds = load_all_dataset()
cover = ds['Baboon']

# Gunakan payload 50 kb (50.000 bit)
payload_bits = np.random.default_rng(359).integers(0, 2, size=50_000, dtype=np.uint8)
res = run_multi_parameter_proposed(cover, payload_bits)
stego = res['stego']

# Hitung selisih mutlak (|Cover - Stego|)
diff = cv2.absdiff(cover, stego)
diff_amplified = np.clip(diff * 50, 0, 255).astype(np.uint8)  # Diperkuat 50x agar terlihat mata

# Simpan gambar perbandingan 3 panel
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cover, cmap='gray')
axes[0].set_title('1. Citra Asli (Cover Image)', fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(stego, cmap='gray')
axes[1].set_title(f"2. Citra Stego (Setelah Disisipi 50 kb)\nPSNR = {res['psnr']:.2f} dB, SSIM = {res['ssim']:.4f}", fontsize=12, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(diff_amplified, cmap='hot')
axes[2].set_title('3. Peta Piksel yang Kena Embed\n(Perubahan diperkuat 50x agar terlihat)', fontsize=12, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
out_path = 'g:/My Drive/kbj/contoh_embedding_visual.png'
plt.savefig(out_path, dpi=300)
plt.close()

# Simpan juga cuplikan matriks zoom-in 5x5 untuk bukti angka
patch_cover = cover[100:105, 100:105]
patch_stego = stego[100:105, 100:105]
patch_diff = patch_stego.astype(int) - patch_cover.astype(int)

np.savetxt('g:/My Drive/kbj/zoom_matrix_cover.txt', patch_cover, fmt='%3d')
np.savetxt('g:/My Drive/kbj/zoom_matrix_stego.txt', patch_stego, fmt='%3d')
np.savetxt('g:/My Drive/kbj/zoom_matrix_diff.txt', patch_diff, fmt='%+2d')

# Salin gambar ke direktori artefak
art_dir = r"C:\Users\Toshiba\.gemini\antigravity\brain\fedacf38-3d04-432c-b118-e30e1b0a9bbb"
shutil.copy(out_path, os.path.join(art_dir, "contoh_embedding_visual.png"))
print("Berhasil membuat visualisasi embedding!")

