#!/usr/bin/env python3
"""
Test runner for Aircraft Dataset Generator

Run all unit tests and verify functionality
"""

import unittest
import sys
import os

# Add the package to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_tests():
    """Run all unit tests"""
    print("🧪 Running Aircraft Dataset Generator Tests")
    print("=" * 50)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
        print(f"📊 Ran {result.testsRun} tests successfully")
        return True
    else:
        print("❌ Some tests failed!")
        print(f"📊 Ran {result.testsRun} tests:")
        print(f"   • Failures: {len(result.failures)}")
        print(f"   • Errors: {len(result.errors)}")
        
        # Print failure details
        if result.failures:
            print("\n🔥 Failures:")
            for test, traceback in result.failures:
                print(f"   • {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if result.errors:
            print("\n💥 Errors:")
            for test, traceback in result.errors:
                print(f"   • {test}: {traceback.split('Error:')[-1].strip()}")
        
        return False


def verify_installation():
    """Verify package can be imported and basic functionality works"""
    print("\n🔍 Verifying Installation")
    print("-" * 30)
    
    try:
        # Test imports
        from aircraft_toolkit import Dataset2D
        from aircraft_toolkit.models.military import F15Fighter, B52Bomber, C130Transport
        print("✅ Package imports successful")
        
        # Test basic model creation
        f15 = F15Fighter()
        assert len(f15.silhouette_points) > 0
        print("✅ Aircraft models functional")
        
        # Test dataset creation (without generation)
        dataset = Dataset2D(aircraft_types=['F15'], num_samples=1, image_size=(64, 64))
        assert 'F15' in dataset.aircraft_models
        print("✅ Dataset creation functional")
        
        print("✅ Installation verification complete")
        return True
        
    except Exception as e:
        print(f"❌ Installation verification failed: {e}")
        return False


def main():
    """Main test runner"""
    print("🛩️  Aircraft Dataset Generator - Test Suite")
    print("=" * 50)
    
    # Verify installation first
    if not verify_installation():
        sys.exit(1)
    
    # Run unit tests
    if not run_tests():
        sys.exit(1)
    
    print("\n🎉 All tests and verifications passed!")
    print("🚀 Aircraft Dataset Generator is ready for use!")


if __name__ == "__main__":
    main()