"""
Tests for the U-Net architecture in src/model.py.
Builds the model with a dummy input — no training or real data required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import build_UNET


def test_unet_output_shape():
    # The model should map (batch, 256, 256, 5) -> (batch, 256, 256, 1)
    model = build_UNET(input_shape=(256, 256, 5))

    assert model.output_shape == (None, 256, 256, 1)


def test_unet_input_shape():
    model = build_UNET(input_shape=(256, 256, 5))

    assert model.input_shape == (None, 256, 256, 5)


def test_unet_custom_output_channels():
    # n_classes parameter should control the number of output channels
    model = build_UNET(input_shape=(128, 128, 3), n_classes=2)

    assert model.output_shape == (None, 128, 128, 2)
