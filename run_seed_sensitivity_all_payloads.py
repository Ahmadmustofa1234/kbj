"""Complete the B_equation6 seed-sensitivity run for all payload sizes.

This script reuses the existing seed 0-29 records for 50 and 100 kb, runs
only the missing payload sizes, and writes new detail and summary CSV files.
It does not modify the baseline implementation or its evaluation CSV.
"""

from __future__ import annotations

import csv
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np

import test_steganography as baseline


ROOT = Path(__file__).resolve().parent
BITS_PER_KB = 1000
SEEDS = tuple(range(30))
PAYLOADS_KB = (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
REUSED_PAYLOADS_KB = {50, 100}
EXISTING_DETAILS = ROOT / "steganography_seed_sensitivity_B_equation6_1000bits_details.csv"
BASELINE_CSV = ROOT / "steganography_evaluation_B_equation6_1000bits.csv"
DETAILS_CSV = ROOT / "steganography_seed_sensitivity_B_equation6_1000bits_all_payloads_details.csv"
SUMMARY_CSV = ROOT / "steganography_seed_sensitivity_B_equation6_1000bits_all_payloads_summary.csv"
RNG_DESCRIPTION = (
    "np.random.default_rng(seed).integers(0, 2, size=100000, dtype=np.uint8)"
)
PAIR_DIFFERENCE = "signed right - center; int16 arithmetic"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_reusable_records() -> dict[tuple[str, int, int], dict[str, str]]:
    records = read_csv(EXISTING_DETAILS)
    selected = [
        row for row in records if int(row["payload_label_kb"]) in REUSED_PAYLOADS_KB
    ]
    expected_count = len(baseline.NATURAL_IMAGES) * len(SEEDS) * len(REUSED_PAYLOADS_KB)
    if len(selected) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} reusable 50/100 kb records, found {len(selected)}"
        )

    reusable = {}
    for row in selected:
        key = (row["image"], int(row["payload_label_kb"]), int(row["seed"]))
        if key in reusable:
            raise RuntimeError(f"Duplicate reusable seed result: {key}")
        if (
            row["capacity_interpretation"] != "B_equation6"
            or row["pair_difference"] != PAIR_DIFFERENCE
            or row["preprocessing"] != "disabled"
            or row["grayscale_conversion"]
            != "cv2.imread(path, cv2.IMREAD_GRAYSCALE)"
            or int(row["bits_per_kb_assumption"]) != BITS_PER_KB
            or int(row["requested_bits"]) != key[1] * BITS_PER_KB
            or int(row["seed"]) not in SEEDS
            or row["rng"] != RNG_DESCRIPTION
            or row["status"] != "ok"
            or row["all_bits_identical"] != "True"
            or float(row["recovery_rate_percent"]) != 100.0
        ):
            raise RuntimeError(f"Reusable record has incompatible settings: {key}")
        reusable[key] = row

    expected_keys = {
        (image_name, payload_kb, seed)
        for image_name, _ in baseline.NATURAL_IMAGES
        for payload_kb in REUSED_PAYLOADS_KB
        for seed in SEEDS
    }
    if reusable.keys() != expected_keys:
        raise RuntimeError("Reusable records do not cover every image/payload/seed")
    return reusable


def run_missing_records(
    reusable: dict[tuple[str, int, int], dict[str, str]],
) -> list[dict]:
    rows = list(reusable.values())
    for image_name, filename in baseline.NATURAL_IMAGES:
        image_path = ROOT / filename
        raw_image = image_path.read_bytes()
        cover = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if cover is None:
            raise FileNotFoundError(image_path)
        baseline.validate_grayscale_image(cover, filename)
        source_hash = hashlib.sha256(raw_image).hexdigest()

        feature_start = time.perf_counter()
        feature_map, pairing_map, _ = baseline.calculate_capacity_components(cover)
        capacity_map = baseline.combine_capacity_maps(
            feature_map, pairing_map, "equation_6"
        )
        feature_time = time.perf_counter() - feature_start
        capacity_bits = int(capacity_map.sum(dtype=np.int64))

        for seed in SEEDS:
            message = np.random.default_rng(seed).integers(
                0, 2, size=max(PAYLOADS_KB) * BITS_PER_KB, dtype=np.uint8
            )
            for payload_kb in PAYLOADS_KB:
                if payload_kb in REUSED_PAYLOADS_KB:
                    continue

                requested_bits = payload_kb * BITS_PER_KB
                secret = message[:requested_bits]
                if requested_bits > capacity_bits:
                    raise RuntimeError(
                        f"{image_name}/{seed}/{payload_kb}kb exceeds capacity"
                    )

                embedding_start = time.perf_counter()
                stego, embedded_bits = baseline.embed_data(
                    cover, secret, capacity_map
                )
                embedding_time = time.perf_counter() - embedding_start
                if embedded_bits != requested_bits:
                    raise RuntimeError(
                        f"Embedded {embedded_bits}/{requested_bits} bits for "
                        f"{image_name}, seed {seed}, {payload_kb} kb"
                    )

                extraction_capacity_start = time.perf_counter()
                extraction_features, extraction_pairing, _ = (
                    baseline.calculate_capacity_components(cover)
                )
                extraction_map = baseline.combine_capacity_maps(
                    extraction_features, extraction_pairing, "equation_6"
                )
                extraction_capacity_time = (
                    time.perf_counter() - extraction_capacity_start
                )
                if not np.array_equal(extraction_map, capacity_map):
                    raise RuntimeError(
                        f"Extraction capacity changed for {image_name}, seed {seed}"
                    )

                extraction_start = time.perf_counter()
                recovered = baseline.extract_bits(
                    stego, extraction_map, requested_bits
                )
                extraction_time = time.perf_counter() - extraction_start
                recovered_array = np.asarray(recovered, dtype=np.uint8)
                matching_bits = int(np.count_nonzero(recovered_array == secret))
                all_identical = (
                    len(recovered) == requested_bits
                    and matching_bits == requested_bits
                )
                recovery_rate = 100.0 * matching_bits / requested_bits
                if not all_identical or recovery_rate != 100.0:
                    raise RuntimeError(
                        f"Recovery mismatch for {image_name}, seed {seed}, "
                        f"{payload_kb} kb"
                    )

                mse, psnr = baseline.calculate_mse_psnr(cover, stego)
                ssim = baseline.calculate_ssim(cover, stego)
                reference_index = baseline.DEFAULT_PAYLOADS_KB.index(payload_kb)
                reference_psnr = baseline.PAPER_PROPOSED_PSNR_DB[image_name][
                    reference_index
                ]
                rows.append(
                    {
                        "image": image_name,
                        "source_file": filename,
                        "source_sha256": source_hash,
                        "grayscale_conversion": (
                            "cv2.imread(path, cv2.IMREAD_GRAYSCALE)"
                        ),
                        "capacity_interpretation": "B_equation6",
                        "pair_difference": PAIR_DIFFERENCE,
                        "preprocessing": "disabled",
                        "seed": seed,
                        "rng": RNG_DESCRIPTION,
                        "payload_label_kb": payload_kb,
                        "bits_per_kb_assumption": BITS_PER_KB,
                        "requested_bits": requested_bits,
                        "embedded_bits": embedded_bits,
                        "recovered_bits": len(recovered),
                        "matching_bits": matching_bits,
                        "recovery_rate_percent": f"{recovery_rate:.6f}",
                        "all_bits_identical": all_identical,
                        "capacity_bits": capacity_bits,
                        "payload_bpp": f"{requested_bits / cover.size:.12f}",
                        "mse": f"{mse:.12f}",
                        "psnr_db": f"{psnr:.9f}",
                        "ssim_data_range_255": f"{ssim:.6f}",
                        "paper_proposed_psnr_db": f"{reference_psnr:.3f}",
                        "psnr_minus_paper_db": f"{psnr - reference_psnr:.9f}",
                        "feature_capacity_calculation_s": f"{feature_time:.9f}",
                        "embedding_s": f"{embedding_time:.9f}",
                        "extraction_capacity_recompute_s": (
                            f"{extraction_capacity_time:.9f}"
                        ),
                        "extraction_bits_s": f"{extraction_time:.9f}",
                        "status": "ok",
                    }
                )
        print(
            f"{image_name}: computed the 9 missing payloads for seeds 0-29; "
            f"reused 50 and 100 kb records"
        )
    return rows


def build_summary(
    details: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
) -> list[dict]:
    detail_index = {
        (row["image"], int(row["payload_label_kb"]), int(row["seed"])): row
        for row in details
    }
    baseline_index = {
        (row["image"], int(row["payload_label_kb"])): row
        for row in baseline_rows
    }
    summary = []
    for image_name, _ in baseline.NATURAL_IMAGES:
        for payload_kb in PAYLOADS_KB:
            runs = [
                detail_index[(image_name, payload_kb, seed)]
                for seed in SEEDS
            ]
            psnr_values = np.asarray(
                [float(row["psnr_db"]) for row in runs], dtype=np.float64
            )
            baseline_row = baseline_index[(image_name, payload_kb)]
            reference_index = baseline.DEFAULT_PAYLOADS_KB.index(payload_kb)
            reference_psnr = baseline.PAPER_PROPOSED_PSNR_DB[image_name][
                reference_index
            ]
            all_identical = all(
                row["all_bits_identical"] in (True, "True")
                and float(row["recovery_rate_percent"]) == 100.0
                and row["status"] == "ok"
                for row in runs
            )
            recovery_rates = np.asarray(
                [float(row["recovery_rate_percent"]) for row in runs],
                dtype=np.float64,
            )
            summary.append(
                {
                    "image": image_name,
                    "payload_label_kb": payload_kb,
                    "bits_per_kb_assumption": BITS_PER_KB,
                    "requested_bits": payload_kb * BITS_PER_KB,
                    "capacity_interpretation": "B_equation6",
                    "pair_difference": PAIR_DIFFERENCE,
                    "grayscale_conversion": (
                        "cv2.imread(path, cv2.IMREAD_GRAYSCALE)"
                    ),
                    "preprocessing": "disabled",
                    "seed_start": min(SEEDS),
                    "seed_end": max(SEEDS),
                    "seed_count": len(SEEDS),
                    "rng": RNG_DESCRIPTION,
                    "paper_proposed_psnr_db": f"{reference_psnr:.3f}",
                    "baseline_seed_42_psnr_db": baseline_row["psnr_db"],
                    "baseline_seed_42_ssim": baseline_row[
                        "ssim_data_range_255"
                    ],
                    "baseline_seed_42_rr_percent": baseline_row[
                        "recovery_rate_percent"
                    ],
                    "baseline_seed_42_all_bits_identical": baseline_row[
                        "all_bits_identical"
                    ],
                    "baseline_seed_42_status": baseline_row["status"],
                    "mean_psnr_db": f"{np.mean(psnr_values):.9f}",
                    "std_psnr_db_sample_ddof1": f"{np.std(psnr_values, ddof=1):.9f}",
                    "min_psnr_db": f"{np.min(psnr_values):.9f}",
                    "max_psnr_db": f"{np.max(psnr_values):.9f}",
                    "paper_reference_within_observed_range": bool(
                        np.min(psnr_values)
                        <= reference_psnr
                        <= np.max(psnr_values)
                    ),
                    "mean_minus_paper_db": (
                        f"{np.mean(psnr_values) - reference_psnr:.9f}"
                    ),
                    "seed_rr_min_percent": f"{np.min(recovery_rates):.6f}",
                    "seed_rr_max_percent": f"{np.max(recovery_rates):.6f}",
                    "all_seed_runs_rr_100_and_identical": all_identical,
                    "all_seed_run_statuses_ok": all(
                        row["status"] == "ok" for row in runs
                    ),
                    "baboon_40kb_table_note": (
                        "Table I visually prints 60.552 dB at both 30 kb and "
                        "40 kb; possible table inconsistency, retained as printed."
                        if image_name == "Baboon" and payload_kb == 40
                        else ""
                    ),
                }
            )
    return summary


def main() -> None:
    if not EXISTING_DETAILS.is_file() or not BASELINE_CSV.is_file():
        raise FileNotFoundError(
            "Required existing seed-50/100 details or protected baseline CSV is missing"
        )
    baseline_hash_before = hashlib.sha256(BASELINE_CSV.read_bytes()).hexdigest()
    reusable = load_reusable_records()
    details = run_missing_records(reusable)

    expected_rows = (
        len(baseline.NATURAL_IMAGES) * len(PAYLOADS_KB) * len(SEEDS)
    )
    if len(details) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} detail rows, got {len(details)}")
    keys = {
        (row["image"], int(row["payload_label_kb"]), int(row["seed"]))
        for row in details
    }
    if len(keys) != expected_rows:
        raise RuntimeError("Duplicate or missing image/payload/seed detail rows")

    baseline_rows = read_csv(BASELINE_CSV)
    summary = build_summary(details, baseline_rows)
    write_csv(DETAILS_CSV, details)
    write_csv(SUMMARY_CSV, summary)
    baseline_hash_after = hashlib.sha256(BASELINE_CSV.read_bytes()).hexdigest()
    if baseline_hash_before != baseline_hash_after:
        raise RuntimeError("Protected baseline CSV changed during the experiment")

    print(f"Details CSV: {DETAILS_CSV.resolve()} ({len(details)} rows)")
    print(f"Summary CSV: {SUMMARY_CSV.resolve()} ({len(summary)} rows)")
    print(f"Baseline CSV unchanged: SHA256 {baseline_hash_after}")
    for row in summary:
        print(
            f"{row['image']},{row['payload_label_kb']}kb,"
            f"seed42={row['baseline_seed_42_psnr_db']},"
            f"mean={row['mean_psnr_db']},"
            f"std={row['std_psnr_db_sample_ddof1']},"
            f"range={row['min_psnr_db']}..{row['max_psnr_db']},"
            f"paper={row['paper_proposed_psnr_db']},"
            f"in_range={row['paper_reference_within_observed_range']},"
            f"RR100_identical={row['all_seed_runs_rr_100_and_identical']}"
        )


if __name__ == "__main__":
    main()
