from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import KDTree

from ._bbox import label_slices
from .typing import Cont, Neig, Node
from .validation import unique_labels, validate_label_image

Pair = tuple[int, int]


def _validate_hole_bridge_options(
    max_hole_area: int | None,
    max_hole_distance: float | None,
    inferred_contact: float,
) -> tuple[int | None, float | None, float]:
    """Validate and normalize options shared by the hole-bridging builders."""
    if max_hole_area is not None:
        if not isinstance(max_hole_area, (int, np.integer)) or max_hole_area < 0:
            raise ValueError("max_hole_area must be a non-negative integer or None")
        max_hole_area = int(max_hole_area)
    if max_hole_distance is not None:
        max_hole_distance = float(max_hole_distance)
        if not np.isfinite(max_hole_distance) or max_hole_distance < 0:
            raise ValueError("max_hole_distance must be finite and non-negative or None")
    inferred_contact = float(inferred_contact)
    if not np.isfinite(inferred_contact) or inferred_contact <= 0:
        raise ValueError("inferred_contact must be finite and positive")
    return max_hole_area, max_hole_distance, inferred_contact


def _hole_facing_pixels(
    labels: np.ndarray,
    hole: np.ndarray,
    *,
    background: int,
) -> dict[int, np.ndarray]:
    """Return hole pixels touching each foreground label by 4-connectivity."""
    facing: dict[int, list[np.ndarray]] = {}

    # Record coordinates on the hole side of each horizontal or vertical
    # boundary. Diagonal-only labels deliberately do not border the hole.
    def acc(hole_side: np.ndarray, label_side: np.ndarray, offset: tuple[int, int]) -> None:
        mask = hole_side & (label_side != background)
        for label in np.unique(label_side[mask]):
            coords = np.argwhere(mask & (label_side == label))
            coords += np.asarray(offset)
            facing.setdefault(int(label), []).append(coords)

    acc(hole[1:, :], labels[:-1, :], (1, 0))
    acc(hole[:-1, :], labels[1:, :], (0, 0))
    acc(hole[:, 1:], labels[:, :-1], (0, 1))
    acc(hole[:, :-1], labels[:, 1:], (0, 0))
    return {label: np.unique(np.concatenate(parts), axis=0) for label, parts in facing.items()}


def _candidate_hole_pairs(
    facing: dict[int, np.ndarray],
    *,
    max_hole_distance: float | None,
) -> set[Pair]:
    """Build the optionally distance-filtered clique around one hole."""
    labels = sorted(facing)
    trees = (
        {label: KDTree(points) for label, points in facing.items()}
        if max_hole_distance is not None
        else {}
    )
    pairs: set[Pair] = set()
    for idx, label_a in enumerate(labels):
        for label_b in labels[idx + 1 :]:
            if max_hole_distance is not None:
                distances, _ = trees[label_a].query(facing[label_b], k=1)
                if float(np.min(distances)) > max_hole_distance:
                    continue
            pairs.add((label_a, label_b))
    return pairs


def _contacts_through_filled_hole(
    labels: np.ndarray,
    hole: np.ndarray,
    *,
    background: int,
    eight: bool,
    diag_weight: float | None,
) -> dict[Pair, float]:
    """Measure contacts introduced by nearest-label filling of one local hole."""
    # These imports are local because preprocessing imports the high-level I/O
    # module, which in turn imports this adjacency module.
    from .preprocessing import fill_internal_gaps_edt

    filled = fill_internal_gaps_edt(labels, background=background, max_distance=None)
    totals: dict[Pair, float] = {}

    def acc(
        a: np.ndarray,
        b: np.ndarray,
        hole_a: np.ndarray,
        hole_b: np.ndarray,
        weight: float,
    ) -> None:
        # Only interfaces created in or immediately against this hole count.
        mask = (hole_a | hole_b) & (a != b) & (a != background) & (b != background)
        if not np.any(mask):
            return
        aa = a[mask].ravel()
        bb = b[mask].ravel()
        pairs = np.stack([np.minimum(aa, bb), np.maximum(aa, bb)], axis=1)
        unique, counts = np.unique(pairs, axis=0, return_counts=True)
        for (label_a, label_b), count in zip(unique, counts, strict=True):
            pair = (int(label_a), int(label_b))
            totals[pair] = totals.get(pair, 0.0) + float(count) * weight

    acc(filled[:-1, :], filled[1:, :], hole[:-1, :], hole[1:, :], 1.0)
    acc(filled[:, :-1], filled[:, 1:], hole[:, :-1], hole[:, 1:], 1.0)
    if eight and labels.shape[0] > 1 and labels.shape[1] > 1:
        diagonal_weight = 1.0 if diag_weight is None else float(diag_weight)
        acc(
            filled[:-1, :-1],
            filled[1:, 1:],
            hole[:-1, :-1],
            hole[1:, 1:],
            diagonal_weight,
        )
        acc(
            filled[:-1, 1:],
            filled[1:, :-1],
            hole[:-1, 1:],
            hole[1:, :-1],
            diagonal_weight,
        )
    return totals


def _hole_pair_contributions(
    labels: np.ndarray,
    *,
    background: int,
    eight: bool,
    diag_weight: float | None,
    max_hole_area: int | None,
    max_hole_distance: float | None,
    inferred_contact: float,
    measure_contacts: bool,
) -> dict[Pair, float]:
    """Return inferred pair contributions accumulated over internal holes."""
    # Keep this import local for the same adjacency/preprocessing I/O cycle
    # described in _contacts_through_filled_hole.
    from .preprocessing import image_has_holes

    has_holes, holes = image_has_holes(labels, background=background)
    if not has_holes:
        return {}

    components, _ = ndi.label(holes, structure=ndi.generate_binary_structure(2, 1)) # type: ignore
    contributions: dict[Pair, float] = {}
    for component, slc in label_slices(components, background=0, padding=1).items():
        local_hole = components[slc] == component
        if max_hole_area is not None and np.count_nonzero(local_hole) > max_hole_area:
            continue
        local_labels = labels[slc]
        facing = _hole_facing_pixels(local_labels, local_hole, background=background)
        candidates = _candidate_hole_pairs(facing, max_hole_distance=max_hole_distance)
        if not candidates:
            continue
        measured = (
            _contacts_through_filled_hole(
                local_labels,
                local_hole,
                background=background,
                eight=eight,
                diag_weight=diag_weight,
            )
            if measure_contacts
            else {}
        )
        for pair in candidates:
            contribution = measured.get(pair, inferred_contact) if measure_contacts else 1.0
            contributions[pair] = contributions.get(pair, 0.0) + contribution
    return contributions


def _neighbors_from_pairs(pairs: np.ndarray) -> Neig:
    """Convert sorted undirected pair rows to a symmetric adjacency mapping."""
    neighbors: dict[int, list[int]] = {}
    for label_a, label_b in pairs:
        neighbors.setdefault(int(label_a), []).append(int(label_b))
        neighbors.setdefault(int(label_b), []).append(int(label_a))
    return {label: np.asarray(values, dtype=np.int64) for label, values in neighbors.items()}


def adjacency_with_unique_from_labels(
    im,
    background=0,
    eight: bool = True,
    allow_background_contacts: bool = False,
    *,
    bridge_holes: bool = False,
    max_hole_area: int | None = None,
    max_hole_distance: float | None = None,
    inferred_contact: float = 1.0,
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
    bridge_holes : bool, optional
        If ``True``, infer possible contacts between labels bordering the same
        internal hole. Default is ``False``.
    max_hole_area : int, optional
        Skip holes containing more than this many pixels.
    max_hole_distance : float, optional
        Maximum distance between hole-facing pixels for an inferred pair.
    inferred_contact : float, optional
        Fallback contact used by weighted graph builders. Accepted here for API
        consistency. Default is ``1.0``.

    Returns
    -------
    neighbors : dict[int, np.ndarray]
        Mapping where ``neighbors[a]`` contains labels touching label ``a``.
    pairs : np.ndarray
        Unique undirected touching-label pairs as an ``(n, 2)`` array. Each row
        is sorted so the smaller label value appears first.
    """
    labels = validate_label_image(im, background=background)
    max_hole_area, max_hole_distance, inferred_contact = _validate_hole_bridge_options(
        max_hole_area,
        max_hole_distance,
        inferred_contact,
    )
    h, w = labels.shape
    pairs_chunks = []
    empty = np.empty((0, 2), dtype=np.int64)

    def acc(a: np.ndarray, b: np.ndarray) -> None:
        mask = (
            a != b
            if allow_background_contacts
            else (a != b) & (a != background) & (b != background)
        )
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
    pairs = np.concatenate(pairs_chunks, axis=0) if pairs_chunks else empty
    if bridge_holes:
        inferred = _hole_pair_contributions(
            labels,
            background=background,
            eight=eight,
            diag_weight=None,
            max_hole_area=max_hole_area,
            max_hole_distance=max_hole_distance,
            inferred_contact=inferred_contact,
            measure_contacts=False,
        )
        if inferred:
            inferred_pairs = np.asarray(list(inferred), dtype=np.int64)
            pairs = np.concatenate([pairs, inferred_pairs], axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    unique_pairs = np.unique(pairs.astype(np.int64), axis=0)
    return _neighbors_from_pairs(unique_pairs), unique_pairs


def adjacency_from_labels(
    im,
    background=0,
    eight: bool = True,
    allow_background_contacts: bool = False,
    *,
    bridge_holes: bool = False,
    max_hole_area: int | None = None,
    max_hole_distance: float | None = None,
    inferred_contact: float = 1.0,
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
    bridge_holes, max_hole_area, max_hole_distance, inferred_contact : optional
        Hole-bridging controls described by
        :func:`adjacency_with_unique_from_labels`.

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
        bridge_holes=bridge_holes,
        max_hole_area=max_hole_area,
        max_hole_distance=max_hole_distance,
        inferred_contact=inferred_contact,
    )
    return neighbors


def adjacency_pairs_from_labels(
    im,
    background=0,
    eight: bool = True,
    allow_background_contacts: bool = False,
    *,
    bridge_holes: bool = False,
    max_hole_area: int | None = None,
    max_hole_distance: float | None = None,
    inferred_contact: float = 1.0,
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
        bridge_holes=bridge_holes,
        max_hole_area=max_hole_area,
        max_hole_distance=max_hole_distance,
        inferred_contact=inferred_contact,
    )
    return pairs


def adjacency_with_contact_from_labels(
    im,
    background=0,
    eight: bool = True,
    diag_weight: float | None = None,
    allow_background_contacts: bool = False,
    *,
    bridge_holes: bool = False,
    max_hole_area: int | None = None,
    max_hole_distance: float | None = None,
    inferred_contact: float = 1.0,
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
    bridge_holes : bool, optional
        If ``True``, infer contacts between labels bordering internal holes.
    max_hole_area : int, optional
        Skip holes containing more than this many pixels. If ``None``, all holes are
        eligible for bridging.
    max_hole_distance : float, optional
        Maximum distance between hole-facing pixels for an inferred pair. If ``None``,
        all hole-facing pairs are eligible for bridging.
    inferred_contact : float, optional
        Contact contribution for inferred pairs not created by local EDT fill.

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
    When ``bridge_holes=True``, locally filled contacts or ``inferred_contact``
    are added for the uncertainty clique around each eligible internal hole.
    """
    labels = validate_label_image(im, background=background)
    max_hole_area, max_hole_distance, inferred_contact = _validate_hole_bridge_options(
        max_hole_area,
        max_hole_distance,
        inferred_contact,
    )
    h, w = labels.shape
    if h == 0 or w == 0:
        return {}, {}
    totals: dict[tuple[int, int], float] = {}

    def acc(a: np.ndarray, b: np.ndarray, weight: float) -> None:
        mask = (
            a != b
            if allow_background_contacts
            else (a != b) & (a != background) & (b != background)
        )
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

    if bridge_holes:
        inferred = _hole_pair_contributions(
            labels,
            background=background,
            eight=eight,
            diag_weight=diag_weight,
            max_hole_area=max_hole_area,
            max_hole_distance=max_hole_distance,
            inferred_contact=inferred_contact,
            measure_contacts=True,
        )
        for pair, contribution in inferred.items():
            totals[pair] = totals.get(pair, 0.0) + contribution

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


def label_pixel_counts(
    labels,
    *,
    background=0,
    include_background: bool = False,
) -> dict[int, int]:
    """
    Count pixels for each label while preserving original label IDs.

    Parameters
    ----------
    labels : np.ndarray
        2-D integer label image.
    background : int, optional
        Background label. Default is ``0``.
    include_background : bool, optional
        If ``False`` (default), omit the background label from the result.

    Returns
    -------
    dict[int, int]
        Mapping ``label -> number of pixels``.
    """
    arr = validate_label_image(labels, background=background)
    values, counts = np.unique(arr, return_counts=True)
    out: dict[int, int] = {}
    for value, count in zip(values, counts, strict=True):
        label = int(value)
        if label == background and not include_background:
            continue
        out[label] = int(count)
    return out


def graph_from_labels(
    labels,
    *,
    background=0,
    eight: bool = True,
    diag_weight: float | None = None,
    allow_background_contacts: bool = False,
    include_centroids: bool = True,
    include_pixel_counts: bool = True,
    bridge_holes: bool = False,
    max_hole_area: int | None = None,
    max_hole_distance: float | None = None,
    inferred_contact: float = 1.0,
) -> tuple[Neig, Cont, dict[int, np.ndarray] | None, dict[int, int] | None]:
    """
    Build adjacency, contact, centroid, and pixel-count graph data from labels.

    Contact values are neighboring pixel-pair counts. They are useful graph
    weights, but they are not exact geometric contact lengths. Hole-bridging
    parameters are forwarded to :func:`adjacency_with_contact_from_labels`.
    """
    neighbors, contacts = adjacency_with_contact_from_labels(
        labels,
        background=background,
        eight=eight,
        diag_weight=diag_weight,
        allow_background_contacts=allow_background_contacts,
        bridge_holes=bridge_holes,
        max_hole_area=max_hole_area,
        max_hole_distance=max_hole_distance,
        inferred_contact=inferred_contact,
    )
    centroids = get_centroids(labels, background=background) if include_centroids else None
    pixel_counts = (
        label_pixel_counts(
            labels,
            background=background,
            include_background=allow_background_contacts,
        )
        if include_pixel_counts
        else None
    )
    return neighbors, contacts, centroids, pixel_counts


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
        [
            int(label)
            for label in neighbors
            if label != background and label_is_border(neighbors, label, background)
        ],
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
