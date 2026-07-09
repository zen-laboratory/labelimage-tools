from __future__ import annotations

import numpy as np

import labelimage_tools as lit


def test_adjacency_expected_pairs_and_contacts(simple_labels):
    neighbors, pairs = lit.adjacency_with_unique_from_labels(simple_labels, eight=False)
    assert {tuple(row) for row in pairs.tolist()} == {(1, 2), (1, 3), (2, 3)}
    assert set(neighbors[1]) == {2, 3}
    neighbors2, contacts = lit.adjacency_with_contact_from_labels(simple_labels, eight=False)
    assert set(neighbors2[1]) == {2, 3}
    assert all(np.all(values > 0) for values in contacts.values())


def test_four_vs_eight_neighborhood_differs():
    labels = np.array([[1, 0], [0, 2]], dtype=np.int64)
    assert lit.adjacency_from_labels(labels, eight=False) == {}
    adj8 = lit.adjacency_from_labels(labels, eight=True)
    assert set(adj8[1]) == {2}


def test_centroids_and_background_border_behavior(simple_labels):
    labels = np.pad(simple_labels, 1)
    neighbors = lit.adjacency_from_labels(labels, allow_background_contacts=True)
    assert lit.label_is_border(neighbors, 1)
    assert set(lit.border_labels(neighbors)) == {1, 2, 3}
    centroids = lit.get_centroids(simple_labels)
    assert np.allclose(centroids[1], [0.5, 1.0])


def test_adjacency_pairs_alias(simple_labels):
    pairs = lit.adjacency_pairs_from_labels(simple_labels)
    assert pairs.shape[1] == 2


def test_label_pixel_counts_preserves_labels_and_background_option():
    labels = np.array(
        [
            [0, 5, 5],
            [10, 10, 5],
            [0, 10, 42],
        ],
        dtype=np.int64,
    )
    assert lit.label_pixel_counts(labels) == {5: 3, 10: 3, 42: 1}
    assert lit.label_pixel_counts(labels, include_background=True) == {0: 2, 5: 3, 10: 3, 42: 1}


def test_graph_from_labels_returns_aligned_graph_data(simple_labels):
    neighbors, contacts, centroids, pixel_counts = lit.graph_from_labels(
        simple_labels,
        eight=False,
    )

    assert centroids is not None
    assert pixel_counts is not None
    assert set(neighbors) == {1, 2, 3}
    assert set(centroids) == {1, 2, 3}
    assert set(pixel_counts) == {1, 2, 3}
    assert pixel_counts[1] == int(np.count_nonzero(simple_labels == 1))

    for label, nbrs in neighbors.items():
        assert label in contacts
        assert len(contacts[label]) == len(nbrs)
        for nbr, contact in zip(nbrs, contacts[label], strict=True):
            reverse_index = list(neighbors[int(nbr)]).index(label)
            assert np.isclose(contact, contacts[int(nbr)][reverse_index])
