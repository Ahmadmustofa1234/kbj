"""Baseline reimplementation of the proposed method in the cited paper.

Paper-grounded details:
* Use grayscale images, row-major center-pixel traversal, and exclude the
  one-pixel border; calculate population standard deviation, range, and
  Shannon entropy on each 3x3 patch.
* Feature thresholds are sigma (10, 25), range (20, 50), and entropy
  (1.2, 2.2); feature cases are all-Low -> 1 bit, at least two-Medium and
  no-High -> 2 bits, exactly one-High -> 1 bit, otherwise skip.
* Fig. 1 gives PairStruct branches: sigma <= 15 and signed D <= 4 -> 1 bit;
  15 < sigma <= 20 and D <= 4 -> 2 bits; 20 < sigma <= 30 and D > 4 ->
  2 bits; otherwise skip. D = right_pixel - center_pixel.
* The written Eq. (6) says max(1, min(feature_bits, pairing_bits)); the
  flowchart instead says min(...) and skips when it is zero. This baseline
  treats either feature- or PairStruct-skip as a skip, then applies Eq. (6)
  to eligible pixels. This keeps the explicit skip branches effective.
* The paper does not specify whether preprocessing was enabled in its tables,
  the filter parameters, or the order of the filters. Baseline runs without
  filtering. Optional bilateral-then-Gaussian filtering is an explicit
  alternative, not a paper-confirmed setting, and requires caller parameters.
* The paper does not define kb as 1000 or 1024 bits. The default experiment
  assumes 1 kb = 1024 bits, matching the prior script, and reports exact bits.
* The paper does not define the order of two bits in a 2-bit pixel slot.
  This implementation places the first payload bit in bit 1 and the second
  in bit 0, and uses the same order during extraction.

Run `python test_steganography.py --help` for options. Dependencies:
numpy, opencv-python, scikit-image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import time
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from skimage.metrics import structural_similarity


DEFAULT_PAYLOADS_KB = (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
DEFAULT_SEED = 42
FEATURE_THRESHOLDS = {
    "std": (10.0, 25.0),
    "range": (20.0, 50.0),
    "entropy": (1.2, 2.2),
}
NATURAL_IMAGES = (
    ("Baboon", "4.2.03.tiff"),
    ("Peppers", "4.2.07.tiff"),
    ("Boat", "boat.512.tiff"),
)

# Values read visually from the PDF's Table I, Proposed column.
PAPER_PROPOSED_PSNR_DB = {
    "Baboon": (75.432, 65.334, 62.379, 60.552, 60.552, 58.362, 57.558,
               56.872, 56.266, 55.780, 55.320),
    "Peppers": (75.165, 65.347, 62.337, 60.559, 59.292, 58.348, 57.567,
                56.875, 56.313, 55.813, 55.330),
    "Boat": (75.258, 65.305, 62.351, 60.567, 59.304, 58.367, 57.525,
             56.851, 56.287, 55.803, 55.345),
}

# The Proposed Baboon values in Table II (not Prior [7]).
PAPER_PROPOSED_BABOON_SSIM = {
    40: 1.000, 50: 1.000, 60: 0.999, 70: 0.999, 80: 0.999,
    90: 0.999, 100: 0.999,
}
PAPER_PROPOSED_BABOON_RR_PERCENT = {kb: 100.0 for kb in PAPER_PROPOSED_BABOON_SSIM}


def validate_grayscale_image(image: np.ndarray, name: str = "image") -> None:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(f"{name} must be a 2D uint8 grayscale image")
    if min(image.shape) < 3:
        raise ValueError(f"{name} must be at least 3x3 pixels")


def preprocess_image(
    image: np.ndarray,
    bilateral_diameter: int,
    bilateral_sigma_color: float,
    bilateral_sigma_space: float,
    gaussian_kernel_size: int,
    gaussian_sigma: float,
) -> np.ndarray:
    """Apply one explicit bilateral-then-Gaussian preprocessing alternative."""
    validate_grayscale_image(image)
    if bilateral_diameter <= 0:
        raise ValueError("bilateral_diameter must be positive")
    if gaussian_kernel_size <= 0 or gaussian_kernel_size % 2 == 0:
        raise ValueError("gaussian_kernel_size must be a positive odd integer")
    bilateral = cv2.bilateralFilter(
        image, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space
    )
    return cv2.GaussianBlur(
        bilateral, (gaussian_kernel_size, gaussian_kernel_size), gaussian_sigma
    )


def _entropy_lookup() -> np.ndarray:
    """Build entropy values for the equality-run patterns of nine sorted pixels."""
    lookup = np.zeros(256, dtype=np.float64)
    for mask in range(256):
        run_lengths: list[int] = []
        current_run = 1
        for bit in range(8):
            if mask & (1 << bit):
                run_lengths.append(current_run)
                current_run = 1
            else:
                current_run += 1
        run_lengths.append(current_run)
        probabilities = np.asarray(run_lengths, dtype=np.float64) / 9.0
        lookup[mask] = -np.sum(probabilities * np.log2(probabilities))
    return lookup


ENTROPY_LOOKUP = _entropy_lookup()
ENTROPY_MASK_WEIGHTS = (1 << np.arange(8, dtype=np.uint16))


def _classify_feature(values: np.ndarray, low_threshold: float, medium_threshold: float):
    low = values < low_threshold
    medium = (values >= low_threshold) & (values <= medium_threshold)
    high = values > medium_threshold
    return low, medium, high


def feature_bit_capacity(
    std_values: np.ndarray, range_values: np.ndarray, entropy_values: np.ndarray
) -> np.ndarray:
    """Apply all feature-case rules, returning 0, 1, or 2 bits per center."""
    feature_values = (
        _classify_feature(std_values, *FEATURE_THRESHOLDS["std"]),
        _classify_feature(range_values, *FEATURE_THRESHOLDS["range"]),
        _classify_feature(entropy_values, *FEATURE_THRESHOLDS["entropy"]),
    )
    low_count = sum(category[0].astype(np.uint8) for category in feature_values)
    medium_count = sum(category[1].astype(np.uint8) for category in feature_values)
    high_count = sum(category[2].astype(np.uint8) for category in feature_values)

    result = np.zeros(np.broadcast_shapes(
        std_values.shape, range_values.shape, entropy_values.shape
    ), dtype=np.uint8)
    result[low_count == 3] = 1
    result[(medium_count >= 2) & (high_count == 0)] = 2
    result[high_count == 1] = 1
    return result


def pairstruct_bit_capacity(
    std_values: np.ndarray, signed_differences: np.ndarray
) -> np.ndarray:
    """Apply every PairStruct branch as drawn in Fig. 1 of the paper."""
    result = np.zeros(np.broadcast_shapes(std_values.shape, signed_differences.shape),
                      dtype=np.uint8)
    first = (std_values <= 15) & (signed_differences <= 4)
    second = (std_values > 15) & (std_values <= 20) & (signed_differences <= 4)
    third = (std_values > 20) & (std_values <= 30) & (signed_differences > 4)
    result[first] = 1
    result[second | third] = 2
    return result


def combine_capacity_maps(
    feature_bits: np.ndarray,
    pairing_bits: np.ndarray,
    interpretation: Literal["min", "equation_6"],
) -> np.ndarray:
    """Combine feature and PairStruct capacities under either paper reading."""
    if feature_bits.shape != pairing_bits.shape:
        raise ValueError("feature_bits and pairing_bits must have identical shapes")
    if not np.isin(feature_bits, (0, 1, 2)).all() or not np.isin(
        pairing_bits, (0, 1, 2)
    ).all():
        raise ValueError("capacity maps may contain only 0, 1, or 2")

    if interpretation == "min":
        return np.minimum(feature_bits, pairing_bits)
    if interpretation == "equation_6":
        result = np.zeros(feature_bits.shape, dtype=np.uint8)
        feature_eligible = feature_bits > 0
        result[feature_eligible] = np.maximum(
            1, np.minimum(feature_bits[feature_eligible], pairing_bits[feature_eligible])
        )
        return result
    raise ValueError(f"unsupported capacity interpretation: {interpretation}")


def calculate_capacity_components(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return feature, PairStruct, and final capacity maps for interior pixels."""
    validate_grayscale_image(image)
    windows = sliding_window_view(image, (3, 3))
    flattened = windows.reshape(*windows.shape[:2], 9)

    std_values = np.std(flattened, axis=-1, dtype=np.float64)
    range_values = np.ptp(flattened, axis=-1)

    sorted_pixels = np.sort(flattened, axis=-1)
    changed = sorted_pixels[..., 1:] != sorted_pixels[..., :-1]
    masks = changed.astype(np.uint16) @ ENTROPY_MASK_WEIGHTS
    entropy_values = ENTROPY_LOOKUP[masks]

    feature_bits = feature_bit_capacity(std_values, range_values, entropy_values)
    center = image[1:-1, 1:-1].astype(np.int16)
    right = image[1:-1, 2:].astype(np.int16)
    signed_differences = right - center
    pairing_bits = pairstruct_bit_capacity(std_values, signed_differences)
    final_bits = combine_capacity_maps(feature_bits, pairing_bits, "min")
    return feature_bits, pairing_bits, final_bits


def calculate_capacity_map(image: np.ndarray) -> np.ndarray:
    """Return row-major capacities for centers (1..h-2, 1..w-2)."""
    return calculate_capacity_components(image)[2]


def capacity_value_counts(capacity_map: np.ndarray) -> dict[int, int]:
    """Count pixels with each supported allocation, including zero capacity."""
    counts = np.bincount(capacity_map.ravel(), minlength=3)
    return {bits: int(counts[bits]) for bits in range(3)}


def validate_payload(secret_bits: Sequence[int]) -> None:
    values = np.asarray(secret_bits)
    if values.ndim != 1 or not np.isin(values, (0, 1)).all():
        raise ValueError("secret_bits must be a one-dimensional sequence of 0/1 bits")


def embed_data(
    cover_image: np.ndarray, secret_bits: Sequence[int], capacity_map: np.ndarray
) -> tuple[np.ndarray, int]:
    """Embed the complete payload, rejecting insufficient capacity beforehand."""
    validate_grayscale_image(cover_image, "cover_image")
    validate_payload(secret_bits)
    expected_map_shape = (cover_image.shape[0] - 2, cover_image.shape[1] - 2)
    if capacity_map.shape != expected_map_shape:
        raise ValueError(f"capacity_map must have shape {expected_map_shape}")
    capacity_bits = int(np.sum(capacity_map, dtype=np.int64))
    if len(secret_bits) > capacity_bits:
        raise ValueError(
            f"payload has {len(secret_bits)} bits, exceeding capacity of {capacity_bits} bits"
        )

    stego = cover_image.copy()
    payload = np.asarray(secret_bits, dtype=np.uint8)
    bit_index = 0
    for center_index, capacity in enumerate(capacity_map.ravel()):
        if bit_index == len(payload):
            break
        if capacity == 0:
            continue
        count = min(int(capacity), len(payload) - bit_index)
        row, column = divmod(center_index, expected_map_shape[1])
        row += 1
        column += 1
        value = int(stego[row, column])
        if count == 1:
            value = (value & ~1) | int(payload[bit_index])
        elif count == 2:
            pair = (int(payload[bit_index]) << 1) | int(payload[bit_index + 1])
            value = (value & ~3) | pair
        else:
            raise ValueError(f"invalid per-pixel capacity: {capacity}")
        stego[row, column] = np.uint8(value)
        bit_index += count

    if bit_index != len(payload):
        raise RuntimeError(f"embedded {bit_index} of {len(payload)} requested bits")
    return stego, bit_index


def extract_bits(
    stego_image: np.ndarray, capacity_map: np.ndarray, payload_size: int
) -> list[int]:
    """Read exactly payload_size bits using a map recalculated from the cover."""
    validate_grayscale_image(stego_image, "stego_image")
    expected_map_shape = (stego_image.shape[0] - 2, stego_image.shape[1] - 2)
    if capacity_map.shape != expected_map_shape:
        raise ValueError(f"capacity_map must have shape {expected_map_shape}")
    if payload_size < 0 or payload_size > int(np.sum(capacity_map, dtype=np.int64)):
        raise ValueError("requested extraction length is outside the capacity map")

    bits: list[int] = []
    for center_index, capacity in enumerate(capacity_map.ravel()):
        if len(bits) == payload_size:
            break
        count = min(int(capacity), payload_size - len(bits))
        row, column = divmod(center_index, expected_map_shape[1])
        value = int(stego_image[row + 1, column + 1])
        if count == 1:
            bits.append(value & 1)
        elif count == 2:
            bits.extend(((value >> 1) & 1, value & 1))
        elif count != 0:
            raise ValueError(f"invalid per-pixel capacity: {capacity}")
    if len(bits) != payload_size:
        raise RuntimeError(f"extracted {len(bits)} of {payload_size} requested bits")
    return bits


def extract_data(
    stego_image: np.ndarray, original_cover: np.ndarray, payload_size: int
) -> list[int]:
    """Recalculate capacity from the original cover, then extract requested bits."""
    capacity_map = calculate_capacity_map(original_cover)
    return extract_bits(stego_image, capacity_map, payload_size)


def calculate_mse_psnr(cover: np.ndarray, stego: np.ndarray) -> tuple[float, float]:
    difference = cover.astype(np.float64) - stego.astype(np.float64)
    mse = float(np.mean(difference * difference, dtype=np.float64))
    psnr = math.inf if mse == 0.0 else 10.0 * math.log10((255.0**2) / mse)
    return mse, psnr


def calculate_ssim(cover: np.ndarray, stego: np.ndarray) -> float:
    return float(structural_similarity(cover, stego, data_range=255))


def load_natural_images(input_dir: Path):
    images = []
    for name, filename in NATURAL_IMAGES:
        path = input_dir / filename
        if not path.is_file():
            print(f"WARNING: missing dataset image: {path}")
            continue
        raw_bytes = path.read_bytes()
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"WARNING: OpenCV could not decode {path}")
            continue
        validate_grayscale_image(image, filename)
        images.append({
            "name": name,
            "path": path,
            "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "source_bytes": len(raw_bytes),
            "image": image,
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "dtype": str(image.dtype),
            "grayscale_conversion": "cv2.imread(..., cv2.IMREAD_GRAYSCALE)",
        })
    return images


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("no evaluation rows were produced")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_payloads(value: str) -> tuple[int, ...]:
    try:
        payloads = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("payloads must be comma-separated integers") from error
    if not payloads or any(payload <= 0 for payload in payloads):
        raise argparse.ArgumentTypeError("all payload sizes must be positive")
    return payloads


def _preprocessing_description(args: argparse.Namespace) -> str:
    if not args.preprocess:
        return "disabled (baseline assumption; paper does not say whether Table I used it)"
    return (
        "bilateral then Gaussian; "
        f"d={args.bilateral_diameter}, sigmaColor={args.bilateral_sigma_color}, "
        f"sigmaSpace={args.bilateral_sigma_space}, "
        f"ksize={args.gaussian_kernel_size}, sigma={args.gaussian_sigma}"
    )


def evaluate(args: argparse.Namespace) -> Path:
    images = load_natural_images(args.input_dir)
    if not images:
        raise FileNotFoundError(
            f"none of the requested USC-SIPI images were found/read in {args.input_dir}"
        )

    if args.preprocess:
        required = (
            args.bilateral_diameter, args.bilateral_sigma_color,
            args.bilateral_sigma_space, args.gaussian_kernel_size, args.gaussian_sigma,
        )
        if any(value is None for value in required):
            raise ValueError(
                "--preprocess requires all bilateral and Gaussian parameter options; "
                "the paper does not specify them"
            )
    preprocessing = _preprocessing_description(args)
    rng = np.random.default_rng(args.seed)
    payload_bits = rng.integers(0, 2, size=max(args.payloads_kb) * args.bits_per_kb,
                                dtype=np.uint8)

    args.stego_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for entry in images:
        source_image = entry["image"]
        if args.preprocess:
            cover = preprocess_image(
                source_image, args.bilateral_diameter, args.bilateral_sigma_color,
                args.bilateral_sigma_space, args.gaussian_kernel_size,
                args.gaussian_sigma,
            )
        else:
            cover = source_image.copy()

        feature_start = time.perf_counter()
        feature_map, pairing_map, _ = calculate_capacity_components(cover)
        capacity_map = combine_capacity_maps(
            feature_map, pairing_map, args.capacity_interpretation
        )
        feature_seconds = time.perf_counter() - feature_start
        capacity_bits = int(np.sum(capacity_map, dtype=np.int64))
        pixel_count = int(cover.size)
        capacity_bpp = capacity_bits / pixel_count
        feature_counts = capacity_value_counts(feature_map)
        pairing_counts = capacity_value_counts(pairing_map)
        final_counts = capacity_value_counts(capacity_map)
        print(
            f"{entry['name']}: {entry['width']}x{entry['height']} {entry['dtype']}, "
            f"capacity={capacity_bits} bits ({capacity_bpp:.8f} bpp), "
            f"feature_bits counts 0/1/2={feature_counts[0]}/{feature_counts[1]}/{feature_counts[2]}, "
            f"pairing_bits counts 0/1/2={pairing_counts[0]}/{pairing_counts[1]}/{pairing_counts[2]}, "
            f"final_bits counts 0/1/2={final_counts[0]}/{final_counts[1]}/{final_counts[2]}, "
            f"feature_time={feature_seconds:.6f}s, SHA256={entry['source_sha256']}"
        )
        if capacity_map.size != (entry["height"] - 2) * (entry["width"] - 2):
            raise RuntimeError("capacity maps must contain exactly the interior pixels")
        if entry["width"] == 512 and entry["height"] == 512 and capacity_map.size != 260_100:
            raise RuntimeError("512x512 images must have 260,100 interior pixels")

        for payload_kb in args.payloads_kb:
            requested_bits = payload_kb * args.bits_per_kb
            secret = payload_bits[:requested_bits]
            row = {
                "image": entry["name"],
                "source_file": entry["path"].name,
                "source_sha256": entry["source_sha256"],
                "source_bytes": entry["source_bytes"],
                "width": entry["width"],
                "height": entry["height"],
                "dtype": entry["dtype"],
                "grayscale_conversion": entry["grayscale_conversion"],
                "preprocessing": preprocessing,
                "seed": args.seed,
                "capacity_interpretation": args.capacity_interpretation,
                "payload_label_kb": payload_kb,
                "bits_per_kb_assumption": args.bits_per_kb,
                "requested_bits": requested_bits,
                "payload_bpp": requested_bits / pixel_count,
                "capacity_bits": capacity_bits,
                "capacity_bpp": capacity_bpp,
                "interior_pixels": int(capacity_map.size),
                "feature_bits_0_pixels": feature_counts[0],
                "feature_bits_1_pixels": feature_counts[1],
                "feature_bits_2_pixels": feature_counts[2],
                "pairing_bits_0_pixels": pairing_counts[0],
                "pairing_bits_1_pixels": pairing_counts[1],
                "pairing_bits_2_pixels": pairing_counts[2],
                "final_bits_0_pixels": final_counts[0],
                "final_bits_1_pixels": final_counts[1],
                "final_bits_2_pixels": final_counts[2],
                "feature_calculation_s": f"{feature_seconds:.9f}",
                "embedding_s": "",
                "extraction_capacity_recompute_s": "",
                "extraction_bits_s": "",
                "extraction_total_s": "",
                "embedded_bits": 0,
                "recovered_bits": 0,
                "matching_bits": 0,
                "recovery_rate_percent": "",
                "all_bits_identical": False,
                "mse": "",
                "psnr_db": "",
                "ssim_data_range_255": "",
                "status": "",
                "stego_png": "",
                "paper_proposed_psnr_db": "",
                "psnr_delta_db": "",
                "paper_proposed_ssim": "",
                "ssim_delta": "",
                "paper_proposed_rr_percent": "",
                "rr_delta_percentage_points": "",
            }

            if requested_bits > capacity_bits:
                row["status"] = "capacity_exceeded_before_embedding"
                rows.append(row)
                print(
                    f"  {payload_kb} kb ({requested_bits} bits): rejected; "
                    f"capacity is {capacity_bits} bits"
                )
                continue

            embed_start = time.perf_counter()
            stego, embedded_bits = embed_data(cover, secret, capacity_map)
            embedding_seconds = time.perf_counter() - embed_start
            if embedded_bits != requested_bits:
                raise RuntimeError(
                    f"embedded {embedded_bits} bits, requested {requested_bits}"
                )

            stem = f"{entry['name'].lower()}_{payload_kb}kb_{requested_bits}bits_seed{args.seed}"
            png_path = args.stego_dir / f"{stem}.png"
            if not cv2.imwrite(str(png_path), stego):
                raise OSError(f"failed to write lossless PNG: {png_path}")
            decoded_stego = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
            if decoded_stego is None or not np.array_equal(decoded_stego, stego):
                raise RuntimeError(f"PNG round-trip changed pixels: {png_path}")

            extraction_feature_start = time.perf_counter()
            extraction_feature_map, extraction_pairing_map, _ = (
                calculate_capacity_components(cover)
            )
            extraction_capacity_map = combine_capacity_maps(
                extraction_feature_map,
                extraction_pairing_map,
                args.capacity_interpretation,
            )
            extraction_feature_seconds = time.perf_counter() - extraction_feature_start
            if not np.array_equal(extraction_capacity_map, capacity_map):
                raise RuntimeError("embedding and extraction capacity maps differ")
            extraction_start = time.perf_counter()
            recovered = extract_bits(decoded_stego, extraction_capacity_map, requested_bits)
            extraction_bits_seconds = time.perf_counter() - extraction_start
            extraction_total_seconds = extraction_feature_seconds + extraction_bits_seconds
            if len(recovered) != requested_bits:
                raise RuntimeError(
                    f"extracted {len(recovered)} bits, requested {requested_bits}"
                )
            matching_bits = int(np.count_nonzero(
                np.asarray(recovered, dtype=np.uint8) == secret
            ))
            recovery_rate = 100.0 * matching_bits / requested_bits
            all_identical = matching_bits == requested_bits

            mse, psnr = calculate_mse_psnr(cover, decoded_stego)
            ssim_value = calculate_ssim(cover, decoded_stego)
            paper_index = DEFAULT_PAYLOADS_KB.index(payload_kb) if payload_kb in DEFAULT_PAYLOADS_KB else None
            paper_psnr = (
                PAPER_PROPOSED_PSNR_DB[entry["name"]][paper_index]
                if paper_index is not None else None
            )
            paper_ssim = (
                PAPER_PROPOSED_BABOON_SSIM.get(payload_kb)
                if entry["name"] == "Baboon" else None
            )
            paper_rr = (
                PAPER_PROPOSED_BABOON_RR_PERCENT.get(payload_kb)
                if entry["name"] == "Baboon" else None
            )
            if not all_identical:
                row["status"] = "extraction_mismatch"
            else:
                row["status"] = "ok"

            row.update({
                "embedding_s": f"{embedding_seconds:.9f}",
                "extraction_capacity_recompute_s": f"{extraction_feature_seconds:.9f}",
                "extraction_bits_s": f"{extraction_bits_seconds:.9f}",
                "extraction_total_s": f"{extraction_total_seconds:.9f}",
                "embedded_bits": embedded_bits,
                "recovered_bits": len(recovered),
                "matching_bits": matching_bits,
                "recovery_rate_percent": f"{recovery_rate:.6f}",
                "all_bits_identical": all_identical,
                "mse": f"{mse:.12f}",
                "psnr_db": f"{psnr:.9f}",
                "ssim_data_range_255": f"{ssim_value:.6f}",
                "stego_png": str(png_path),
                "paper_proposed_psnr_db": (
                    f"{paper_psnr:.3f}" if paper_psnr is not None else ""
                ),
                "psnr_delta_db": (
                    f"{psnr - paper_psnr:.9f}" if paper_psnr is not None else ""
                ),
                "paper_proposed_ssim": (
                    f"{paper_ssim:.3f}" if paper_ssim is not None else ""
                ),
                "ssim_delta": (
                    f"{ssim_value - paper_ssim:.6f}" if paper_ssim is not None else ""
                ),
                "paper_proposed_rr_percent": (
                    f"{paper_rr:.6f}" if paper_rr is not None else ""
                ),
                "rr_delta_percentage_points": (
                    f"{recovery_rate - paper_rr:.6f}" if paper_rr is not None else ""
                ),
            })
            rows.append(row)
            print(
                f"  {payload_kb} kb ({requested_bits} bits): {row['status']}, "
                f"PSNR={psnr:.6f} dB, SSIM={ssim_value:.6f}, "
                f"RR={recovery_rate:.6f}%, "
                f"embed={embedding_seconds:.6f}s, extract={extraction_total_seconds:.6f}s"
            )

    _write_csv(args.output, rows)
    print(f"CSV: {args.output.resolve()}")
    print(f"Stego PNG directory: {args.stego_dir.resolve()}")
    print(f"Seed: {args.seed}; payload assumption: 1 kb = {args.bits_per_kb} bits")
    print(f"Capacity interpretation: {args.capacity_interpretation}")
    print(f"Preprocessing: {preprocessing}")
    return args.output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--input-dir", type=Path, default=script_dir)
    parser.add_argument("--output", type=Path, default=script_dir / "steganography_evaluation.csv")
    parser.add_argument("--stego-dir", type=Path, default=script_dir / "stego_png")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--capacity-interpretation",
        choices=("min", "equation_6"),
        default="min",
        help="combine feature and PairStruct maps using flowchart min or Eq. (6)",
    )
    parser.add_argument(
        "--payloads-kb", type=_parse_payloads,
        default=DEFAULT_PAYLOADS_KB,
        help="nominal payload sizes; default: 1,10,20,...,100",
    )
    parser.add_argument(
        "--bits-per-kb", type=int, default=1024,
        choices=(1000, 1024),
        help="explicit conversion assumption for nominal kb labels (default: 1024 bits)",
    )
    parser.add_argument(
        "--preprocess", action="store_true",
        help="enable bilateral then Gaussian filtering (paper parameters are unspecified)",
    )
    parser.add_argument("--bilateral-diameter", type=int)
    parser.add_argument("--bilateral-sigma-color", type=float)
    parser.add_argument("--bilateral-sigma-space", type=float)
    parser.add_argument("--gaussian-kernel-size", type=int)
    parser.add_argument("--gaussian-sigma", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bits_per_kb <= 0:
        raise SystemExit("--bits-per-kb must be a positive integer")
    if args.preprocess:
        try:
            evaluate(args)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
