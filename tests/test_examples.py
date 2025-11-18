"""Tests for thrml_th example scripts."""

from __future__ import annotations

from examples import blockwise_sampler, ising_sampler, pytorch_training_loop


def test_ising_sampler_runs(capsys):
    """Test that ising_sampler example runs with minimal workload."""
    exit_code = ising_sampler.main(
        [
            "--length",
            "8",
            "--n-warmup",
            "2",
            "--n-samples",
            "4",
            "--steps-per-sample",
            "1",
            "--chains",
            "2",
            "--show-samples",
            "1",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Collected" in captured.out
    assert "overall magnetization" in captured.out
    assert "First 1 samples" in captured.out


def test_blockwise_sampler_runs(capsys):
    """Test that blockwise_sampler example runs and reports both partitions."""

    exit_code = blockwise_sampler.main(
        [
            "--length",
            "8",
            "--n-warmup",
            "2",
            "--n-samples",
            "4",
            "--chains",
            "1",
            "--show-samples",
            "0",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Partition: contiguous halves" in captured.out
    assert "Partition: checkerboard" in captured.out


def test_pytorch_training_loop_runs(capsys):
    """Test that pytorch_training_loop example runs with minimal workload."""
    exit_code = pytorch_training_loop.main(
        [
            "--length",
            "16",
            "--dataset-size",
            "64",
            "--batch-samples",
            "16",
            "--epochs",
            "2",
            "--batch-size",
            "16",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "epoch 01 | loss" in captured.out
    assert "Final accuracy on generated dataset" in captured.out


def test_ising_sampler_shape_output(capsys):
    """Test that ising_sampler produces expected tensor shape."""
    ising_sampler.main(
        [
            "--length",
            "10",
            "--n-samples",
            "5",
            "--n-warmup",
            "1",
            "--steps-per-sample",
            "1",
            "--show-samples",
            "2",
        ]
    )
    captured = capsys.readouterr()
    assert "spin count = 10" in captured.out
    assert "First 2 samples" in captured.out


def test_suppress_warnings_flag():
    """Test that suppress-warnings flag is accepted."""
    exit_code = ising_sampler.main(
        [
            "--length",
            "4",
            "--n-warmup",
            "1",
            "--n-samples",
            "2",
            "--steps-per-sample",
            "1",
            "--no-suppress-warnings",
        ]
    )
    assert exit_code == 0

    exit_code = pytorch_training_loop.main(
        [
            "--length",
            "8",
            "--dataset-size",
            "32",
            "--batch-samples",
            "8",
            "--epochs",
            "1",
            "--no-suppress-warnings",
        ]
    )
    assert exit_code == 0
