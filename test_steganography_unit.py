import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import test_steganography as stego


class CapacityRuleTests(unittest.TestCase):
    def test_feature_case_rules_and_threshold_boundaries(self):
        std = np.array([9.0, 10.0, 26.0, 9.0, 26.0])
        value_range = np.array([19.0, 20.0, 20.0, 20.0, 51.0])
        entropy = np.array([1.1, 1.2, 1.1, 1.1, 2.3])
        np.testing.assert_array_equal(
            stego.feature_bit_capacity(std, value_range, entropy),
            np.array([1, 2, 1, 0, 0], dtype=np.uint8),
        )

    def test_all_pairstruct_branches_and_boundaries(self):
        std = np.array([15.0, 15.01, 20.0, 20.01, 30.0, 30.01])
        difference = np.array([4.0, 4.0, 4.0, 5.0, 5.0, 5.0])
        np.testing.assert_array_equal(
            stego.pairstruct_bit_capacity(std, difference),
            np.array([1, 2, 2, 2, 2, 0], dtype=np.uint8),
        )

    def test_capacity_combination_interpretations_are_distinct(self):
        feature = np.array([[0, 1, 1, 2, 2]], dtype=np.uint8)
        pairing = np.array([[2, 0, 1, 0, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(
            stego.combine_capacity_maps(feature, pairing, "min"),
            np.array([[0, 0, 1, 0, 2]], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            stego.combine_capacity_maps(feature, pairing, "equation_6"),
            np.array([[0, 1, 1, 1, 2]], dtype=np.uint8),
        )

    def test_capacity_map_excludes_the_border(self):
        image = np.zeros((5, 7), dtype=np.uint8)
        capacity = stego.calculate_capacity_map(image)
        self.assertEqual(capacity.shape, (3, 5))
        np.testing.assert_array_equal(capacity, np.ones((3, 5), dtype=np.uint8))

    def test_capacity_histogram_covers_all_interior_pixels(self):
        image = np.zeros((512, 512), dtype=np.uint8)
        feature, pairing, final = stego.calculate_capacity_components(image)
        self.assertEqual(feature.size, 260_100)
        self.assertEqual(pairing.size, 260_100)
        self.assertEqual(final.size, 260_100)
        self.assertEqual(stego.capacity_value_counts(feature), {0: 0, 1: 260_100, 2: 0})
        self.assertEqual(stego.capacity_value_counts(pairing), {0: 0, 1: 260_100, 2: 0})
        self.assertEqual(stego.capacity_value_counts(final), {0: 0, 1: 260_100, 2: 0})


class EmbeddingExtractionTests(unittest.TestCase):
    def test_one_remaining_bit_at_two_bit_location(self):
        cover = np.zeros((3, 3), dtype=np.uint8)
        cover[1, 1] = 10
        capacity = np.array([[2]], dtype=np.uint8)
        stego_image, embedded = stego.embed_data(cover, [1], capacity)
        self.assertEqual(embedded, 1)
        self.assertEqual(int(stego_image[1, 1]) & 1, 1)
        self.assertEqual(stego.extract_bits(stego_image, capacity, 1), [1])

    def test_two_bits_at_two_bit_location_keep_msb_first_order(self):
        cover = np.zeros((3, 3), dtype=np.uint8)
        capacity = np.array([[2]], dtype=np.uint8)
        stego_image, embedded = stego.embed_data(cover, [1, 0], capacity)
        self.assertEqual(embedded, 2)
        self.assertEqual(int(stego_image[1, 1]) & 3, 2)
        self.assertEqual(stego.extract_bits(stego_image, capacity, 2), [1, 0])

    def test_oversize_payload_is_rejected_before_embedding(self):
        cover = np.zeros((3, 3), dtype=np.uint8)
        capacity = np.array([[1]], dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "exceeding capacity"):
            stego.embed_data(cover, [0, 1], capacity)
        np.testing.assert_array_equal(cover, np.zeros((3, 3), dtype=np.uint8))

    def test_png_round_trip_and_extraction_recalculate_from_cover(self):
        cover = np.zeros((16, 16), dtype=np.uint8)
        payload = [1, 0, 1, 1, 0, 0, 1]
        capacity = stego.calculate_capacity_map(cover)
        stego_image, embedded = stego.embed_data(cover, payload, capacity)
        self.assertEqual(embedded, len(payload))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roundtrip.png"
            self.assertTrue(cv2.imwrite(str(path), stego_image))
            decoded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            np.testing.assert_array_equal(decoded, stego_image)
            self.assertEqual(stego.extract_data(decoded, cover, len(payload)), payload)


if __name__ == "__main__":
    unittest.main()
