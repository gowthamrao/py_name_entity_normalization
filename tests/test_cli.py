"""
Tests for the Command-Line Interface (CLI).
"""
import runpy
from unittest.mock import MagicMock

from py_name_entity_normalization.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_cli_build_index(mocker):
    """
    Tests the build-index command in a success scenario.
    """
    # Arrange
    mock_builder_class = mocker.patch("py_name_entity_normalization.cli.IndexBuilder")
    mock_builder_instance = MagicMock()
    mock_builder_class.return_value = mock_builder_instance
    mocker.patch("py_name_entity_normalization.cli.get_session")

    # Create a dummy file to pass the 'exists=True' check
    with runner.isolated_filesystem():
        with open("test.csv", "w") as f:
            f.write("id,name\n1,test")

        # Act
        result = runner.invoke(app, ["build-index", "test.csv", "--force"])

        # Assert
        assert result.exit_code == 0
        assert "Starting Index Build" in result.stdout
        assert "Index Build Successful" in result.stdout
        # Check that the builder was initialized and called correctly
        mock_builder_class.assert_called_once()
        mock_builder_instance.build_index_from_csv.assert_called_once()

        # Verify arguments manually for robustness against path resolution
        args, kwargs = mock_builder_instance.build_index_from_csv.call_args
        assert str(kwargs["csv_path"]).endswith("test.csv")
        assert kwargs["force"] is True


def test_cli_build_index_file_not_found(mocker):
    """
    Tests that build-index fails if the file doesn't exist.
    """
    # Act
    result = runner.invoke(app, ["build-index", "nonexistent.csv"])

    # Assert
    assert result.exit_code != 0
    # The error now comes from Pathlib, not Typer's pre-check
    assert "Error during index build" in result.stdout


def test_cli_verify_index_success(mocker):
    """
    Tests the verify-index command in a success scenario.
    """
    # Arrange
    mock_engine_class = mocker.patch(
        "py_name_entity_normalization.cli.NormalizationEngine"
    )
    mock_engine_instance = MagicMock()
    mock_engine_instance.embedder.get_model_name.return_value = "verified-model"
    mock_engine_class.return_value = mock_engine_instance

    # Act
    result = runner.invoke(app, ["verify-index"])

    # Assert
    assert result.exit_code == 0
    assert "Verifying Index" in result.stdout
    assert "Model consistency check passed" in result.stdout
    assert "verified-model" in result.stdout
    mock_engine_class.assert_called_once()


def test_cli_verify_index_failure(mocker):
    """
    Tests that verify-index handles errors during engine initialization.
    """
    # Arrange
    mock_engine_class = mocker.patch(
        "py_name_entity_normalization.cli.NormalizationEngine"
    )
    mock_engine_class.side_effect = ValueError("Test Error")

    # Act
    result = runner.invoke(app, ["verify-index"])

    # Assert
    assert result.exit_code == 1
    assert "Error during index verification" in result.stdout
    assert "Test Error" in result.stdout


def test_cli_main_entrypoint(mocker):
    """
    Tests the `if __name__ == '__main__'` block.
    """
    # To test the __main__ block, we can't use the CliRunner.
    # Instead, we use runpy to execute the module as a script.
    # We mock the __call__ method of the Typer app object itself
    # to prevent the CLI from actually running, and just assert it was called.
    mock_typer_call = mocker.patch("typer.main.Typer.__call__")
    runpy.run_module("py_name_entity_normalization.cli", run_name="__main__")
    mock_typer_call.assert_called_once()
