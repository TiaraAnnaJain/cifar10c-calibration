"""Smoke test: the package installs and imports."""

import cifar10c_calibration


def test_package_imports() -> None:
    assert cifar10c_calibration.__name__ == "cifar10c_calibration"
