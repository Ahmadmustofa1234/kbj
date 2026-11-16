import numpy as np
from fix import load_all_dataset, run_pairstruct_muzli

ds = load_all_dataset()
for name in ['Baboon', 'Head']:
    cover = ds[name]
    print(f"=== {name} ===")
    for kb in [40, 50, 60, 70, 80, 90, 100]:
        secret = np.random.default_rng(42).integers(0, 2, size=kb*1000, dtype=np.uint8)
        res = run_pairstruct_muzli(cover, secret)
        print(f"{kb} kb: PSNR = {res['psnr']:.3f} dB, SSIM = {res['ssim']:.4f}, RR = {res['rr']:.1f}%")

