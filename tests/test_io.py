from __future__ import annotations

import numpy as np

import labelimage_tools as lit


def test_load_label_image_preserves_integer_labels(sample_path):
    labels = lit.load_label_image(sample_path)
    assert labels.ndim == 2
    assert np.issubdtype(labels.dtype, np.integer)
    assert labels.max() > 0
    assert np.array_equal(labels, lit.load_img(sample_path))


def _assert_neighbors_equal(left, right):
    assert set(left) == set(right)
    for label in left:
        assert set(left[label]) == set(right[label])


def _assert_contacts_equal(left_neighbors, right_neighbors, left, right):
    assert set(left) == set(right)
    for label, nbrs in left_neighbors.items():
        left_map = {int(nbr): float(value) for nbr, value in zip(nbrs, left[label], strict=True)}
        right_map = {
            int(nbr): float(value)
            for nbr, value in zip(right_neighbors[label], right[label], strict=True)
        }
        assert left_map == right_map


def _assert_graph_roundtrip(path, labels):
    neighbors, contacts, centroids, pixel_counts = lit.graph_from_labels(labels, eight=False)
    lit.save_label_graph(
        path,
        neighbors,
        contacts=contacts,
        centroids=centroids,
        pixel_counts=pixel_counts,
        source_image="synthetic",
    )

    loaded = lit.load_label_graph(path)

    _assert_neighbors_equal(neighbors, loaded.neighbors)
    assert loaded.contacts is not None
    _assert_contacts_equal(neighbors, loaded.neighbors, contacts, loaded.contacts)
    assert loaded.centroids is not None
    assert set(loaded.centroids) == set(centroids)
    for label in centroids:
        assert np.allclose(loaded.centroids[label], centroids[label])
    assert loaded.pixel_counts == pixel_counts
    assert loaded.metadata["source_image"] == "synthetic"


def test_label_graph_npz_roundtrip(tmp_path):
    labels = np.array([[1, 1, 2], [1, 3, 2], [3, 3, 2]], dtype=np.int64)
    _assert_graph_roundtrip(tmp_path / "graph.npz", labels)


def test_label_graph_json_roundtrip(tmp_path):
    labels = np.array([[1, 1, 2], [1, 3, 2], [3, 3, 2]], dtype=np.int64)
    _assert_graph_roundtrip(tmp_path / "graph.json", labels)


def test_label_graph_format_inference_and_unknown_suffix(tmp_path):
    labels = np.array([[1, 2], [3, 3]], dtype=np.int64)
    lit.save_label_graph_from_labels(tmp_path / "graph.npz", labels)
    lit.save_label_graph_from_labels(tmp_path / "graph.json", labels)
    assert lit.load_label_graph(tmp_path / "graph.npz").neighbors
    assert lit.load_label_graph(tmp_path / "graph.json").neighbors
    with np.testing.assert_raises_regex(ValueError, "could not infer graph format"):
        lit.save_label_graph_from_labels(tmp_path / "graph.unknown", labels)


def test_save_label_graph_from_labels_preserves_construction_metadata(tmp_path):
    labels = np.array([[1, 1, 2], [1, 3, 2], [3, 3, 2]], dtype=np.int64)
    path = tmp_path / "from_labels.npz"

    lit.save_label_graph_from_labels(
        path,
        labels,
        eight=False,
        source_image="labels.tif",
    )
    loaded = lit.load_label_graph(path)

    assert loaded.contacts is not None
    assert loaded.centroids is not None
    assert loaded.pixel_counts == lit.label_pixel_counts(labels)
    assert loaded.metadata["eight"] is False
    assert loaded.metadata["source_image"] == "labels.tif"
