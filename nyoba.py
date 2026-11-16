import os
import time
import csv
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

from eksperimen_sobel import (
    BASE_DIR,
    load_all_dataset,
    compute_multi_parameter_capacity,
    run_multi_parameter_proposed,
    PAPER_MATCHED_SEEDS,
)

LOW = 30
HIGH = 100
ORDER = ['Baboon', 'Peppers', 'Boat', 'Hand', 'Leg', 'Head', 'Abdominal', 'Chest']
PAYLOAD_KB = [10, 50, 100]


def stable_plan(image, guided=False):
    stable = np.bitwise_and(image, np.uint8(0xFC))
    capacity = compute_multi_parameter_capacity(stable).astype(np.uint8)
    gx = cv2.Sobel(stable, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(stable, cv2.CV_64F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)[1:-1, 1:-1].ravel()

    # Pengaruh Sobel pada kapasitas: gradien rendah/tinggi maks. 1 bit.
    cap = capacity.copy()
    non_medium = (gradient < LOW) | (gradient > HIGH)
    cap[non_medium] = np.minimum(cap[non_medium], 1)
    idx = np.flatnonzero(cap > 0)
    if guided:
        # Prioritas: gradien sedang, rendah, tinggi. Urutan raster dipertahankan
        # di dalam tiap kategori (stable sort) untuk reprodusibilitas.
        category = np.where(gradient[idx] < LOW, 1,
                            np.where(gradient[idx] <= HIGH, 0, 2))
        idx = idx[np.argsort(category, kind='stable')]
    return cap, idx


def embed(image, message, guided=False):
    """Adaptive LSB; ekstraksi dilakukan terpisah dari stego saja."""
    message = np.asarray(message, dtype=np.uint8)
    cap, order = stable_plan(image, guided)
    available = int(cap.sum())
    if len(message) > available:
        raise ValueError(f'Payload {len(message)} bit melebihi kapasitas {available} bit')
    stego = image.copy()
    interior = stego[1:-1, 1:-1].reshape(-1).copy()
    pos = 0
    for idx in order:
        if pos == len(message):
            break
        n = min(int(cap[idx]), len(message)-pos)
        if n == 1:
            interior[idx] = (int(interior[idx]) & 0xFE) | int(message[pos])
        else:
            interior[idx] = (int(interior[idx]) & 0xFC) | (int(message[pos]) << 1) | int(message[pos+1])
        pos += n
    stego[1:-1, 1:-1] = interior.reshape(image.shape[0]-2, image.shape[1]-2)
    return stego, available


def extract_blind(stego, nbits, guided=False):
    """Tidak menerima cover; hanya stego, panjang pesan, dan aturan metode."""
    cap, order = stable_plan(stego, guided)
    interior = stego[1:-1, 1:-1].ravel()
    bits = np.empty(nbits, dtype=np.uint8)
    pos = 0
    for idx in order:
        if pos == nbits:
            break
        n = min(int(cap[idx]), nbits-pos)
        val = int(interior[idx])
        if n == 1:
            bits[pos] = val & 1
        else:
            bits[pos] = (val >> 1) & 1
            bits[pos+1] = val & 1
        pos += n
    if pos != nbits:
        raise ValueError('Ekstraksi gagal: kapasitas tidak cukup')
    return bits


def metrics(cover, stego):
    mse = np.mean((cover.astype(np.float64)-stego.astype(np.float64))**2)
    psnr = float('inf') if mse == 0 else 10*np.log10(255**2/mse)
    return psnr, float(ssim(cover, stego, data_range=255))


def main():
    started = time.time()
    images = load_all_dataset()
    rows = []
    for name in ORDER:
        if name not in images:
            print(f'SKIP {name}: dataset tidak ditemukan')
            continue
        cover = images[name]
        print(f'\n=== {name} ===')
        for kb in PAYLOAD_KB:
            nbits = kb*1000
            seed = PAPER_MATCHED_SEEDS[name][kb]
            msg = np.random.default_rng(seed).integers(0,2,size=nbits,dtype=np.uint8)
            # A: baseline asli dari paper, ekstraksi cover-assisted.
            baseline = run_multi_parameter_proposed(cover, msg)
            a_psnr, a_ssim = baseline['psnr'], baseline['ssim']
            rows.append(dict(image=name,payload_bits=nbits,method='A_baseline_original',
                capacity_bits='',embedded_bits=baseline['embedded_bits'],
                psnr_db=a_psnr,ssim=a_ssim,recovery_pct=baseline['rr'],
                extraction='cover-assisted',full_payload=baseline['embedded_bits']==nbits))
            scores = {'A':a_ssim}
            for code, guided in [('B_4param_raster',False),('C_4param_guided',True)]:
                try:
                    stego, capacity = embed(cover,msg,guided)
                    # PNG roundtrip untuk memastikan ekstraksi dari file citra stego.
                    ok, png = cv2.imencode('.png', stego)
                    if not ok:
                        raise RuntimeError('PNG encode gagal')
                    reloaded = cv2.imdecode(png,cv2.IMREAD_GRAYSCALE)
                    recovered = extract_blind(reloaded,nbits,guided)
                    rr = 100*float(np.mean(recovered==msg))
                    psnr,ssim_score = metrics(cover,reloaded)
                    if rr != 100.0:
                        print(f'PERINGATAN: {code}, RR={rr:.4f}%')
                    rows.append(dict(image=name,payload_bits=nbits,method=code,
                        capacity_bits=capacity,embedded_bits=nbits,psnr_db=psnr,
                        ssim=ssim_score,recovery_pct=rr,extraction='blind PNG',full_payload=True))
                    scores[code[0]] = ssim_score
                except ValueError as exc:
                    print(f'  {kb} kb {code}: {exc}')
                    rows.append(dict(image=name,payload_bits=nbits,method=code,
                        capacity_bits='',embedded_bits=0,psnr_db='',ssim='',
                        recovery_pct='',extraction='blind PNG',full_payload=False))
            print(f'{kb:3} kb | SSIM A={scores["A"]:.9f} B={scores.get("B",float("nan")):.9f} C={scores.get("C",float("nan")):.9f}')
    if rows:
        path = os.path.join(BASE_DIR,'hasil_ablasi_sobel_guided.csv')
        with open(path,'w',newline='',encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nHasil tersimpan: {path}')
    print(f'Waktu: {time.time()-started:.1f} detik')

if __name__ == '__main__':
    main()