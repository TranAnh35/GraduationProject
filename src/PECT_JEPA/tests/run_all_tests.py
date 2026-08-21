"""
PECT-JEPA v0.2 Test Suite Runner (implement.md, Section 4).
Executes the 6 mandatory verification unit tests:
1. test_shape.py (Tensor shape consistency [1, 300, 300, 16] -> [1, 37, 37, 16, 128])
2. test_mask.py (Frame-by-frame block integrity & disjointness)
3. test_information_leakage.py (No target content leakage)
4. test_gradient.py (Gradient flow & Target Encoder isolation)
5. test_overfit.py (Monotonic loss decrease on 1 clip)
6. test_full_scale_forward.py (Full-scale clip benchmark 300x300x16)
"""

import unittest
import sys
import os

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.PECT_JEPA.tests.test_shape import TestShapes
from src.PECT_JEPA.tests.test_mask import TestMask
from src.PECT_JEPA.tests.test_information_leakage import TestInformationLeakage
from src.PECT_JEPA.tests.test_gradient import TestGradient
from src.PECT_JEPA.tests.test_overfit import TestOverfit
from src.PECT_JEPA.tests.test_full_scale_forward import TestFullScaleForward


def run_tests():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    suite.addTests(loader.loadTestsFromTestCase(TestShapes))
    suite.addTests(loader.loadTestsFromTestCase(TestMask))
    suite.addTests(loader.loadTestsFromTestCase(TestInformationLeakage))
    suite.addTests(loader.loadTestsFromTestCase(TestGradient))
    suite.addTests(loader.loadTestsFromTestCase(TestOverfit))
    suite.addTests(loader.loadTestsFromTestCase(TestFullScaleForward))

    print("=" * 70)
    print("RUNNING PECT-JEPA v0.2 UNIT TEST SUITE (implement.md)")
    print("=" * 70)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 70)
    if result.wasSuccessful():
        print("ALL 6 TESTS PASSED SUCCESSFULLY! PECT-JEPA v0.2 refactoring verified.")
    else:
        print(f"TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors.")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
