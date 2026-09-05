import random

import numpy as np
import pytest
from PIL import Image

from roadsigns.image_tools.geometry import clip_boxes, letterbox, padded_tile, windows
from roadsigns.training.mosaic import scale_for, stitch


def test_full_4k_coverage_and_small_sign_preservation():
    tiles = list(windows(4032, 3024))
    assert len(tiles) == len(set(tiles)) == 12
    assert max(r[2] for r in tiles) == 4032
    assert max(r[3] for r in tiles) == 3024
    box = np.array([[1018.0, 1018.0, 1040.0, 1040.0]])
    assert any(np.array_equal(clip_boxes(box, roi)[0], box) for roi in tiles)


def test_small_images_are_padded_without_resizing():
    image = Image.new("RGB", (300, 200), (20, 40, 60))
    tile = padded_tile(image, next(windows(*image.size)))
    assert tile.size == (1280, 1280)
    assert tile.getpixel((299, 199)) == (20, 40, 60)
    assert tile.getpixel((300, 200)) == (114, 114, 114)


def test_crop_area_and_visibility_thresholds():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 9, 10], [-40, 0, 10, 10], [-41, 0, 10, 10]], float)
    clipped, keep = clip_boxes(boxes, (0, 0, 1280, 1280))
    assert keep.tolist() == [True, False, True, False]
    assert clipped.tolist() == [[0, 0, 10, 10], [0, 0, 10, 10]]


def test_letterbox_uses_actual_rounded_dimensions_and_integer_padding():
    image = Image.new("RGB", (4031, 3023))
    box = np.array([[0.0, 0.0, 4031, 3023]])
    canvas, transformed, geometry = letterbox(image, box)
    scale_x, scale_y = geometry["scale_xy"]
    pad_x, pad_y = geometry["pad_xy"]
    recovered = (transformed - [pad_x, pad_y, pad_x, pad_y]) / [scale_x, scale_y, scale_x, scale_y]
    np.testing.assert_allclose(recovered, box)
    assert canvas.size == (1280, 1280)
    assert isinstance(pad_y, int)


def test_tiny_boxes_are_never_shrunk_by_mosaic():
    tiny = np.array([[1000.0, 1000.0, 1022.0, 1022.0]])
    large = np.array([[1000.0, 1000.0, 1100.0, 1100.0]])
    assert all(scale_for(tiny, random.Random(i)) >= 1 for i in range(100))
    assert any(scale_for(large, random.Random(i)) < 1 for i in range(100))


def test_mosaic_pixels_and_labels_share_the_same_transform():
    image = np.full((1280, 1280, 3), 10, dtype=np.uint8)
    boxes = np.array(
        [[1000, 1000, 1022, 1022], [620, 1000, 642, 1022], [632, 1000, 654, 1022]], float
    )
    sample = (image, np.array([[1], [2], [3]]), boxes)
    empty = (np.full_like(image, 20), np.empty((0, 1)), np.empty((0, 4)))
    canvas, classes, clipped, trace = stitch(
        [sample, empty, empty, empty], scale=0, center=(640, 640)
    )
    assert trace["scales"] == [1, 1, 1, 1]
    assert classes.flatten().tolist() == [1]
    assert clipped.tolist() == [[360, 360, 382, 382]]
    assert (canvas[400, 400] == 10).all()
    assert (canvas[400, 700] == 20).all()


@pytest.mark.parametrize("size,overlap", [(0, 0.2), (1280, 1), (1280, -0.1)])
def test_invalid_tile_geometry_fails_at_input(size, overlap):
    with pytest.raises(ValueError):
        list(windows(2000, 1000, size, overlap))
