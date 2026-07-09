from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .typing import Cont, Neig, Node
from .validation import unique_labels, validate_label_image


def adjacency_with_unique_from_labels(
    im,
    background=0,
    eight: bool = True,
    allow_background_contacts: bool = False,
) -> tuple[Neig, np.ndarray]:
    """
    Compute adjacency between labels by scanning neighboring pixels.

    Parameters
    ----------
    im : np.ndarray
        2-D integer array representing a labeled map.
    background : int, optional
        Label value to treat as background. Default is ``0``.
    eight : bool, optional
        If ``True`` (default), include diagonal contacts and therefore use an
        8-neighborhood. If ``False``, use only vertical and horizontal
        4-neighborhood contacts.
    allow_background_contacts : bool, optional
        If ``False`` (default), pairs involving ``background`` are excluded. If
        ``True``, contacts with background are included in the returned adjacency
        and pair array.

    Returns
    -------
    neighbors : dict[int, np.ndarray]
        Mapping where ``neighbors[a]`` contains labels touching label ``a``.
    pairs : np.ndarray
        Unique undirected touching-label pairs as an ``(n, 2)`` array. Each row
        is sorted so the smaller label value appears first.
    """
    labels = validate_label_image(im, background=background)
    h, w = labels.shape
    pairs_chunks = []
    empty = np.empty((0, 2), dtype=np.int64)

    def acc(a: np.ndarray, b: np.ndarray) -> None:
        mask = a != b if allow_background_contacts else (a != b) & (a != background) & (b != background)
        if np.any(mask):
            aa = a[mask].ravel()
            bb = b[mask].ravel()
            pairs_chunks.append(np.stack([np.minimum(aa, bb), np.maximum(aa, bb)], axis=1))

    if h == 0 or w == 0:
        return {}, empty
    acc(labels[:-1, :], labels[1:, :])
    acc(labels[:, :-1], labels[:, 1:])
    if eight and h > 1 and w > 1:
        acc(labels[:-1, :-1], labels[1:, 1:])
        acc(labels[:-1, 1:], labels[1:, :-1])
    if not pairs_chunks:
        return {}, empty

    pairs = np.concatenate(pairs_chunks, axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    uniq = np.unique(pairs.astype(np.int64), axis=0)
    adj: dict[int, list[int]] = {}
    for a, b in uniq:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    return {k: np.asarray(v, dtype=np.int64) for k, v in adj.items()}, uniq


def adjacency_from_labels(
    im,
    background=0,
    eight: bool = True,
    allow_background_contacts: bool = False,
) -> Neig:
    """
    Return a label adjacency mapping for a 2-D labeled image.

    This is a convenience wrapper around
    :func:`adjacency_with_unique_from_labels` that discards the unique pair
    array.

    Parameters
    ----------
    im : np.ndarray
        2-D integer label image.
    background : int, optional
        Background label. Default is ``0``.
    eight : bool, optional
        Whether to include diagonal contacts.
    allow_background_contacts : bool, optional
        Whether to include contacts with the background label.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping ``label -> neighboring labels``.
    """
    neighbors, _ = adjacency_with_unique_from_labels(
        im,
        background=background,
        eight=eight,
        allow_background_contacts=allow_background_contacts,
    )
    return neighbors


def adjacency_pairs_from_labels(
    im,
    background=0,
    eight: bool = True,
    allow_background_contacts: bool = False,
) -> np.ndarray:
    """
    Return unique undirected touching-label pairs.

    Parameters are the same as :func:`adjacency_with_unique_from_labels`.

    Returns
    -------
    np.ndarray
        ``(n, 2)`` integer array of sorted label pairs.
    """
    _, pairs = adjacency_with_unique_from_labels(
        im,
        background=background,
        eight=eight,
        allow_background_contacts=allow_background_contacts,
    )
    return pairs


def adjacency_with_contact_from_labels(
    im,
    background=0,
    eight: bool = True,
    diag_weight: float | None = None,
    allow_background_contacts: bool = False,
) -> tuple[Neig, Cont]:
    """
    Compute adjacency and pixel-neighborhood contact counts.

    Parameters
    ----------
    im : np.ndarray
        2-D integer label image.
    background : int, optional
        Label to treat as background. Default is ``0``.
    eight : bool, optional
        Whether to consider diagonal contacts in addition to vertical and
        horizontal contacts.
    diag_weight : float, optional
        Weight assigned to diagonal contacts when ``eight=True``. If ``None``,
        diagonal contacts count as ``1.0``.
    allow_background_contacts : bool, optional
        If ``True``, contacts involving the background label are counted.

    Returns
    -------
    neighbors : dict[int, np.ndarray]
        Mapping ``label -> neighboring labels``.
    contacts : dict[int, np.ndarray]
        Mapping ``label -> contact counts``. For each label, the contact array is
        aligned with the corresponding neighbors array.

    Notes
    -----
    Contact values are counts of neighboring pixel pairs. They are useful graph
    weights, but they are not guaranteed to be exact geometric contact lengths,
    especially when borders are thick or diagonal contacts are included.
    """
    labels = validate_label_image(im, background=background)
    h, w = labels.shape
    if h == 0 or w == 0:
        return {}, {}
    totals: dict[tuple[int, int], float] = {}

    def acc(a: np.ndarray, b: np.ndarray, weight: float) -> None:
        mask = a != b if allow_background_contacts else (a != b) & (a != background) & (b != background)
        if not np.any(mask):
            return
        aa = a[mask].ravel()
        bb = b[mask].ravel()
        pairs = np.stack([np.minimum(aa, bb), np.maximum(aa, bb)], axis=1)
        uniq, counts = np.unique(pairs, axis=0, return_counts=True)
        for (pa, pb), count in zip(uniq, counts, strict=True):
            key = (int(pa), int(pb))
            totals[key] = totals.get(key, 0.0) + float(count) * weight

    acc(labels[:-1, :], labels[1:, :], 1.0)
    acc(labels[:, :-1], labels[:, 1:], 1.0)
    if eight and h > 1 and w > 1:
        wdiag = 1.0 if diag_weight is None else float(diag_weight)
        acc(labels[:-1, :-1], labels[1:, 1:], wdiag)
        acc(labels[:-1, 1:], labels[1:, :-1], wdiag)

    adj: dict[int, list[int]] = {}
    cont: dict[int, list[float]] = {}
    for (a, b), count in totals.items():
        adj.setdefault(a, []).append(b)
        cont.setdefault(a, []).append(count)
        adj.setdefault(b, []).append(a)
        cont.setdefault(b, []).append(count)
    return (
        {k: np.asarray(v, dtype=np.int64) for k, v in adj.items()},
        {k: np.asarray(cont[k]) for k in cont},
    )


def label_is_border(neighbors: Neig, label: Node, background: Node = 0) -> bool:
    """
    Determine whether a label touches the background.

    Parameters
    ----------
    neighbors : dict
        Adjacency mapping, usually computed with
        ``allow_background_contacts=True``.
    label : int
        Label to check.
    background : int, optional
        Background label. Default is ``0``.

    Returns
    -------
    bool
        ``True`` if ``background`` is among ``label``'s neighbors.
    """
    return bool(background in neighbors.get(label, np.array([], dtype=np.int64)))


def border_labels(neighbors: Neig, background: Node = 0) -> np.ndarray:
    """
    Return all labels that touch the background in an adjacency mapping.

    Parameters
    ----------
    neighbors : dict
        Adjacency mapping that includes background contacts.
    background : int, optional
        Background label. Default is ``0``.

    Returns
    -------
    np.ndarray
        Integer array of labels for which :func:`label_is_border` is true.
    """
    return np.asarray(
        [int(label) for label in neighbors if label != background and label_is_border(neighbors, label, background)],
        dtype=np.int64,
    )


def get_centroids(im, background=0) -> dict[int, np.ndarray]:
    """
    Compute center-of-mass centroids for each non-background label.

    Parameters
    ----------
    im : np.ndarray
        2-D integer label image.
    background : int, optional
        Background label to exclude. Default is ``0``.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping ``label -> centroid`` where centroids are floating-point image
        coordinates in ``(y, x)`` order.
    """
    labels = validate_label_image(im, background=background)
    values = unique_labels(labels, background=background)
    cms = ndi.center_of_mass(np.ones_like(labels), labels=labels, index=values)
    return {int(value): np.asarray(cm, dtype=float) for value, cm in zip(values, cms, strict=True)}


centroids_from_labels = get_centroids
