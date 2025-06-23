"""Basic validation tests that don't import project dependencies."""
import sys
from pathlib import Path

import pytest


class TestBasicSetupValidation:
    """Basic validation tests to ensure the testing infrastructure is properly configured."""

    def test_python_version(self):
        """Verify Python version is 3.8 or higher."""
        assert sys.version_info >= (3, 8), "Python 3.8+ is required"

    def test_pytest_installed(self):
        """Verify pytest is installed and importable."""
        import pytest
        assert pytest.__version__

    def test_pytest_cov_installed(self):
        """Verify pytest-cov is installed and importable."""
        import pytest_cov
        assert pytest_cov

    def test_pytest_mock_installed(self):
        """Verify pytest-mock is installed and importable."""
        import pytest_mock
        assert pytest_mock

    def test_project_structure(self):
        """Verify the expected project structure exists."""
        project_root = Path(__file__).parent.parent
        assert project_root.exists()
        assert (project_root / "src").exists()
        assert (project_root / "tests").exists()
        assert (project_root / "pyproject.toml").exists()

    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit test marker works."""
        assert True

    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration test marker works."""
        assert True

    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow test marker works."""
        assert True

    def test_temp_dir_fixture(self, temp_dir):
        """Test that temp_dir fixture works correctly."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")
        assert test_file.read_text() == "test"

    def test_mock_config_fixture(self, mock_config):
        """Test that mock_config fixture provides expected structure."""
        assert "model" in mock_config
        assert "training" in mock_config
        assert "data" in mock_config
        assert mock_config["model"]["name"] == "test_model"