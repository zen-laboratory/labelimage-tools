from __future__ import annotations

import warnings

import numpy as np
import pytest

import labelimage_tools as lit
import labelimage_tools._optional as optional
import labelimage_tools.junctions as junctions_module


def four_label_meeting():
    return np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ],
        dtype=np.int64,
    )


def test_junction_pixels_and_clusters():
    labels = four_label_meeting()
    mask, labels_at_pixel = lit.junction_pixels_with_labels(labels)
    assert mask.any()
    assert labels_at_pixel
    label_image, junctions = lit.cluster_junctions_with_labels(mask, labels_at_pixel)
    assert label_image.max() == len(junctions)
    assert junctions[0].id == 1
    assert junctions[0].labels == frozenset({1, 2, 3, 4})
    assert np.isfinite(junctions[0].yx).all()
    assert np.issubdtype(junctions[0].pixel_coords.dtype, np.integer)


def test_junction_pixels_falls_back_without_numba(monkeypatch):
    labels = four_label_meeting()
    real_import_module = optional.import_module

    def fake_import_module(module_name):
        if module_name == "numba":
            raise ImportError("numba hidden for test")
        return real_import_module(module_name)

    monkeypatch.setattr(optional, "import_module", fake_import_module)
    monkeypatch.setattr(junctions_module, "_numba_checked", False)
    monkeypatch.setattr(junctions_module, "_numba_junction_mask_core", None)

    mask, labels_at_pixel = lit.junction_pixels_with_labels(labels)

    assert mask.any()
    assert labels_at_pixel
    assert junctions_module._numba_checked is True
    assert junctions_module._numba_junction_mask_core is None


def test_junction_pixels_warns_on_large_image_without_numba(monkeypatch):
    labels = four_label_meeting()
    real_import_module = optional.import_module

    def fake_import_module(module_name):
        if module_name == "numba":
            raise ImportError("numba hidden for test")
        return real_import_module(module_name)

    monkeypatch.setattr(optional, "import_module", fake_import_module)
    monkeypatch.setattr(junctions_module, "_numba_checked", False)
    monkeypatch.setattr(junctions_module, "_numba_junction_mask_core", None)
    monkeypatch.setattr(junctions_module, "JUNCTION_NUMBA_WARNING_MIN_PIXELS", 1)

    with pytest.warns(RuntimeWarning, match="pure-Python fallback"):
        lit.junction_pixels_with_labels(labels)


def test_junction_pixels_warning_can_be_suppressed(monkeypatch):
    labels = four_label_meeting()
    real_import_module = optional.import_module

    def fake_import_module(module_name):
        if module_name == "numba":
            raise ImportError("numba hidden for test")
        return real_import_module(module_name)

    monkeypatch.setattr(optional, "import_module", fake_import_module)
    monkeypatch.setattr(junctions_module, "_numba_checked", False)
    monkeypatch.setattr(junctions_module, "_numba_junction_mask_core", None)
    monkeypatch.setattr(junctions_module, "JUNCTION_NUMBA_WARNING_MIN_PIXELS", 1)

    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        lit.junction_pixels_with_labels(labels, warn_without_numba=False)

    assert not warnings_record


def test_junction_pixels_numba_matches_python_fallback(monkeypatch):
    pytest.importorskip("numba")
    labels = four_label_meeting()

    real_import_module = optional.import_module
    monkeypatch.setattr(junctions_module, "_numba_checked", False)
    monkeypatch.setattr(junctions_module, "_numba_junction_mask_core", None)
    mask_numba, labels_numba = lit.junction_pixels_with_labels(labels)

    def fake_import_module(module_name):
        if module_name == "numba":
            raise ImportError("numba hidden for test")
        return real_import_module(module_name)

    monkeypatch.setattr(optional, "import_module", fake_import_module)
    monkeypatch.setattr(junctions_module, "_numba_checked", False)
    monkeypatch.setattr(junctions_module, "_numba_junction_mask_core", None)
    mask_python, labels_python = lit.junction_pixels_with_labels(labels)

    assert np.array_equal(mask_numba, mask_python)
    assert labels_numba.keys() == labels_python.keys()
    for key in labels_numba:
        assert np.array_equal(labels_numba[key], labels_python[key])


def test_cluster_junctions_with_labels_single_component():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    labels_at_pixel = {2 * 5 + 2: np.array([1, 2, 3])}

    label_image, junctions = lit.cluster_junctions_with_labels(mask, labels_at_pixel)

    assert len(junctions) == 1
    assert label_image[2, 2] == junctions[0].id
    assert junctions[0].id == 1
    assert junctions[0].yx.tolist() == [2.0, 2.0]
    assert np.array_equal(junctions[0].pixel_coords, np.array([[2, 2]]))
    assert junctions[0].labels == frozenset({1, 2, 3})


def test_cluster_junctions_with_labels_multi_component_custom_start_id():
    mask = np.zeros((7, 7), dtype=bool)
    mask[1, 1] = True
    mask[5, 5] = True
    labels_at_pixel = {
        1 * 7 + 1: np.array([1, 2, 3]),
        5 * 7 + 5: np.array([4, 5, 6]),
    }

    label_image, junctions = lit.cluster_junctions_with_labels(
        mask,
        labels_at_pixel,
        start_id=10,
    )

    assert [junction.id for junction in junctions] == [10, 11]
    assert label_image[1, 1] == 10
    assert label_image[5, 5] == 11
    assert set(np.unique(label_image)) == {0, 10, 11}
    assert np.array_equal(junctions[0].pixel_coords, np.array([[1, 1]]))
    assert np.array_equal(junctions[1].pixel_coords, np.array([[5, 5]]))


def test_cluster_junctions_with_labels_multi_pixel_component_unions_labels():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2:4, 1:3] = True
    labels_at_pixel = {
        2 * 5 + 1: np.array([1, 2]),
        2 * 5 + 2: np.array([2, 3]),
        3 * 5 + 1: np.array([3, 4]),
        3 * 5 + 2: np.array([4, 5]),
    }

    label_image, junctions = lit.cluster_junctions_with_labels(mask, labels_at_pixel)

    assert len(junctions) == 1
    assert np.all(label_image[2:4, 1:3] == junctions[0].id)
    assert junctions[0].yx.tolist() == [2.5, 1.5]
    assert set(map(tuple, junctions[0].pixel_coords)) == {(2, 1), (2, 2), (3, 1), (3, 2)}
    assert junctions[0].labels == frozenset({1, 2, 3, 4, 5})


def test_cluster_junctions_with_labels_respects_connectivity():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True
    labels_at_pixel = {
        1 * 4 + 1: np.array([1, 2, 3]),
        2 * 4 + 2: np.array([4, 5, 6]),
    }

    label_image_4, junctions_4 = lit.cluster_junctions_with_labels(
        mask,
        labels_at_pixel,
        connectivity=1,
    )
    label_image_8, junctions_8 = lit.cluster_junctions_with_labels(
        mask,
        labels_at_pixel,
        connectivity=2,
    )

    assert [junction.id for junction in junctions_4] == [1, 2]
    assert label_image_4[1, 1] != label_image_4[2, 2]
    assert len(junctions_8) == 1
    assert label_image_8[1, 1] == label_image_8[2, 2] == junctions_8[0].id
    assert junctions_8[0].labels == frozenset({1, 2, 3, 4, 5, 6})


def test_junctions_from_three_label_meeting_excluding_background():
    labels = np.array([[1, 1, 2], [1, 3, 2], [3, 3, 2]], dtype=np.int64)
    label_image, junctions = lit.junctions_from_labels(labels, background=0)
    assert label_image.max() >= 1
    assert any(j.labels == frozenset({1, 2, 3}) for j in junctions)


def test_merge_close_junctions():
    j1 = lit.Junction(1, np.array([0.0, 0.0]), np.array([[0, 0]]), frozenset({1, 2, 3}))
    j2 = lit.Junction(2, np.array([0.2, 0.2]), np.array([[0, 1]]), frozenset({3, 4, 5}))
    merged = lit.merge_close_junctions([j1, j2], epsilon=1.0)
    assert len(merged) == 1
    assert merged[0].labels == frozenset({1, 2, 3, 4, 5})
