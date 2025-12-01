#!/usr/bin/env python3
"""Test runner for aircraft dataset generator."""
import subprocess
import sys


def main():
    print("🧪 Running Aircraft Dataset Generator Tests...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"], check=False
    )
    return result.returncode


if __name__ == "__main__":
    exit(main())
