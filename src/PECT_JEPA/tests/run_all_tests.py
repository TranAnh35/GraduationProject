"""
PECT-JEPA Test Suite Runner.
Runs all verification tests specified in Section 22:
- 22.1 Shape test
- 22.2 Mask test
- 22.3 Gradient & EMA test
- 22.4 Small Overfit test
- 22.5 Information Leakage test
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
from src.PECT_JEPA.tests.test_gradient import TestGradient
from src.PECT_JEPA.tests.test_information_leakage import TestInformationLeakage
from src.PECT_JEPA.tests.test_overfit import TestOverfit
from src.PECT_JEPA.tests.test_real_tdms import TestRealTDMS


def run_tests():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    suite.addTests(loader.loadTestsFromTestCase(TestShapes))
    suite.addTests(loader.loadTestsFromTestCase(TestMask))
    suite.addTests(loader.loadTestsFromTestCase(TestGradient))
    suite.addTests(loader.loadTestsFromTestCase(TestInformationLeakage))
    suite.addTests(loader.loadTestsFromTestCase(TestOverfit))
    suite.addTests(loader.loadTestsFromTestCase(TestRealTDMS))

    print("=" * 70)
    print("RUNNING PECT-JEPA VERIFICATION TEST SUITE (Section 22)")
    print("=" * 70)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 70)
    if result.wasSuccessful():
        print("ALL TESTS PASSED SUCCESSFULLY! Ready for PECT-JEPA training.")
    else:
        print(f"TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors.")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
