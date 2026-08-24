"""
1D Temporal PECT-JEPA Test Suite Runner (implement.md, Section 5).
Executes the 5 mandatory verification unit tests for 1D Temporal PECT-JEPA:
1. test_shape_1d.py (Tensor shape consistency [B, 500] -> [B, 16, 128])
2. test_mask_1d.py (Late-decay & random patch disjointness)
3. test_leakage_1d.py (Zero target content leakage to Predictor)
4. test_gradient_1d.py (Gradient flow & Target Encoder isolation)
5. test_overfit_1d.py (Monotonic loss decrease on fixed batch)
"""

import unittest
import sys
import os

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.PECT_JEPA.temporal_1d.tests.test_shape_1d import TestShape1D
from src.PECT_JEPA.temporal_1d.tests.test_mask_1d import TestMask1D
from src.PECT_JEPA.temporal_1d.tests.test_leakage_1d import TestLeakage1D
from src.PECT_JEPA.temporal_1d.tests.test_gradient_1d import TestGradient1D
from src.PECT_JEPA.temporal_1d.tests.test_overfit_1d import TestOverfit1D


def run_tests():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    suite.addTests(loader.loadTestsFromTestCase(TestShape1D))
    suite.addTests(loader.loadTestsFromTestCase(TestMask1D))
    suite.addTests(loader.loadTestsFromTestCase(TestLeakage1D))
    suite.addTests(loader.loadTestsFromTestCase(TestGradient1D))
    suite.addTests(loader.loadTestsFromTestCase(TestOverfit1D))

    print("=" * 70)
    print("RUNNING 1D TEMPORAL PECT-JEPA (TS-JEPA) TEST SUITE (implement.md)")
    print("=" * 70)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 70)
    if result.wasSuccessful():
        print("ALL 5 1D UNIT TESTS PASSED SUCCESSFULLY! 1D TS-JEPA implementation verified.")
    else:
        print(f"TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors.")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
