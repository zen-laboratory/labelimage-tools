from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import KDTree

from ._optional import optional_import
from .validation import validate_label_image


def _component_label():
    measure = optional_import(
        "skimage.measure",
        extra="plot",
        feature="Junction clustering",
        package_name="scikit-image",
    )
    return measure.label


_numba_junction_mask_core = None
_numba_checked = False


def _get_numba_junction_mask_core():  # pragma: no cover - exercised via public wrapper
    global _numba_checked, _numba_junction_mask_core
    if _numba_checked:
        return _numba_junction_mask_core
    _numba_checked = True
    try:
        numba = import_module("numba")
    except ImportError:
        return None
    njit = numba.njit
    prange = numba.prange

    @njit(cache=True, parallel=True)
    def _kernel(padded, h, w, min_labels):
        mask = np.zeros((h, w), np.uint8)
        for y in prange(h):
            yy = y + 1
            for x in range(w):
                vals = np.empty(9, padded.dtype)
                nunique = 0
                done = False
                for dy in range(3):
                    if done:
                        break
                    for dx in range(3):
                        value = padded[yy - 1 + dy, x + dx]
                        seen = False
                        for k in range(nunique):
                            if vals[k] == value:
                                seen = True
                                break
                        if not seen:
                            vals[nunique] = value
                            nunique += 1
                            if nunique >= min_labels:
                                mask[y, x] = 1
                                done = True
                                break
        return mask

    _numba_junction_mask_core = _kernel
    return _numba_junction_mask_core


@dataclass(frozen=True)
class Junction:
    """
    Clustered junction in a labeled image.

    Attributes
    ----------
    id : int
        Junction identifier. IDs are 1-based by default so ``0`` can remain
        "no junction" in a junction-label image.
    yx : np.ndarray
        Floating-point subpixel centroid of the junction cluster in image
        coordinates ``(y, x)``.
    pixel_coords : np.ndarray
        Integer pixel coordinates belonging to the junction cluster, with shape
        ``(n_pixels, 2)`` and coordinate order ``(y, x)``.
    labels : frozenset[int]
        Set of cell/region labels observed around the junction pixels.
    """

    id: int
    yx: np.ndarray
    pixel_coords: np.ndarray
    labels: frozenset[int]


def _python_junction_mask_core(padded: np.ndarray, h: int, w: int, min_labels: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        yy = y + 1
        for x in range(w):
            if np.unique(padded[yy - 1 : yy + 2, x : x + 3]).size >= min_labels:
                mask[y, x] = 1
    return mask


def junction_pixels_with_labels(
    labels,
    *,
    background=None,
    min_labels: int = 3,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """
    Find junction pixels and record labels visible at each such pixel.

    A junction pixel is any pixel whose padded 3×3 neighborhood contains at
    least ``min_labels`` distinct labels. This preserves the behavior of
    ``vertexify.find_junctions._junction_pixels_with_labels``.

    Parameters
    ----------
    labels : np.ndarray
        2-D integer label image.
    background : int, optional
        If provided, this label is removed from the neighborhood before labels
        are counted. If ``None`` (default), background participates in the count,
        matching the source implementation.
    min_labels : int, optional
        Minimum number of distinct labels required to mark a pixel as a junction.
        Default is ``3``.

    Returns
    -------
    junction_mask : np.ndarray
        Boolean mask with ``True`` at junction pixels.
    labels_at_pixel : dict[int, np.ndarray]
        Mapping from flattened pixel index ``y * width + x`` to the sorted unique
        labels observed in that pixel's 3×3 neighborhood.
    """
    labels = validate_label_image(labels, background=0 if background is None else background)
    h, w = labels.shape
    padded = np.pad(labels, 1, mode="edge")
    numba_kernel = _get_numba_junction_mask_core()
    if background is None and numba_kernel is not None and min_labels <= 9:
        mask = numba_kernel(padded, h, w, int(min_labels)).astype(bool)
    else:
        mask = np.zeros((h, w), dtype=bool)
        for y in range(h):
            yy = y + 1
            for x in range(w):
                vals = np.unique(padded[yy - 1 : yy + 2, x : x + 3])
                if background is not None:
                    vals = vals[vals != background]
                if vals.size >= min_labels:
                    mask[y, x] = True

    labels_at_pixel: dict[int, np.ndarray] = {}
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs, strict=True):
        yy = y + 1
        vals = np.unique(padded[yy - 1 : yy + 2, x : x + 3])
        if background is not None:
            vals = vals[vals != background]
        if vals.size >= min_labels:
            labels_at_pixel[int(y) * w + int(x)] = vals.astype(np.int64)
    return mask, labels_at_pixel


def cluster_junctions_with_labels(
    junction_mask,
    labels_at_pixel: dict[int, np.ndarray],
    *,
    connectivity: int = 2,
    start_id: int = 1,
) -> tuple[np.ndarray, list[Junction]]:
    """
    Cluster junction pixels into geometric junction objects.

    Connected components of ``junction_mask`` become individual junctions. The
    centroid of each component is the mean of its member pixel coordinates, and
    the junction's label set is the union of the label sets recorded for those
    pixels.

    Parameters
    ----------
    junction_mask : np.ndarray
        Boolean mask of junction pixels.
    labels_at_pixel : dict[int, np.ndarray]
        Mapping produced by :func:`junction_pixels_with_labels`.
    connectivity : int, optional
        Connectivity passed to ``skimage.measure.label``. In 2-D, ``2`` gives
        8-connectivity and is the default.
    start_id : int, optional
        First junction ID. Default is ``1``.

    Returns
    -------
    junction_label_image : np.ndarray
        Integer image where each junction component is labeled by its junction
        ID and non-junction pixels are ``0``.
    junctions : list[Junction]
        Clustered junction objects. Each object has an ID matching
        ``junction_label_image``, a mean ``(y, x)`` coordinate, all member pixel
        coordinates, and the union of labels observed around those pixels.
    """
    mask = np.asarray(junction_mask, dtype=bool)
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.int64), []

    # Component labels are dense positive integers, so scipy's object-finding
    # helper gives one local bounding box per component in component-id order.
    component_labels = _component_label()(mask, connectivity=connectivity)
    objects = ndi.find_objects(component_labels)

    h, w = mask.shape
    junction_label_image = np.zeros(mask.shape, dtype=np.int64)
    junctions: list[Junction] = []

    for component_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue

        # Restrict the component mask lookup to its crop rather than scanning
        # the full image once per component.
        sub = component_labels[slc] # type: ignore
        local_y, local_x = np.nonzero(sub == component_id)
        if local_y.size == 0:
            continue

        y = local_y + slc[0].start
        x = local_x + slc[1].start
        coords = np.column_stack([y, x]).astype(np.int64)

        # IDs remain compact even if scipy ever returns an empty/None slot.
        jid = start_id + len(junctions)
        junction_label_image[y, x] = jid

        # Union the labels recorded around each pixel in the component.
        label_set: set[int] = set()
        for yy, xx in coords:
            vals = labels_at_pixel.get(int(yy) * w + int(xx))
            if vals is not None:
                label_set.update(int(value) for value in vals)

        junctions.append(
            Junction(
                id=jid,
                yx=coords.mean(axis=0).astype(float),
                pixel_coords=coords,
                labels=frozenset(label_set),
            )
        )
    return junction_label_image, junctions


def merge_close_junctions(
    junctions,
    *,
    epsilon: float,
    start_id: int = 1,
) -> list[Junction]:
    """
    Merge junctions closer than ``epsilon`` pixels.

    Parameters
    ----------
    junctions : iterable[Junction]
        Junctions to merge.
    epsilon : float
        Distance threshold in pixels. Junctions whose centroids are within this
        distance are grouped together.
    start_id : int, optional
        First ID assigned to the merged junction list.

    Returns
    -------
    list[Junction]
        Merged junctions. Coordinates are averaged, pixel coordinates are
        concatenated, and label sets are unioned within each merged group.

    Notes
    -----
    This adapts the source ``_merge_close_vertices`` behavior to return public
    :class:`Junction` objects rather than parallel coordinate/label arrays.
    """
    junctions = list(junctions)
    if epsilon <= 0 or len(junctions) <= 1:
        return [
            Junction(
                start_id + i,
                np.asarray(j.yx, dtype=float),
                j.pixel_coords,
                frozenset(j.labels),
            )
            for i, j in enumerate(junctions)
        ]
    positions = np.asarray([j.yx for j in junctions], dtype=float)
    tree = KDTree(positions)
    pairs = tree.query_pairs(epsilon)
    parent = np.arange(len(junctions))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        union(a, b)
    groups: dict[int, list[int]] = {}
    for i in range(len(junctions)):
        groups.setdefault(find(i), []).append(i)
    merged = []
    for offset, group in enumerate(groups.values()):
        coords = np.vstack([junctions[i].pixel_coords for i in group])
        labs = frozenset().union(*(junctions[i].labels for i in group))
        merged.append(
            Junction(
                id=start_id + offset,
                yx=positions[group].mean(axis=0),
                pixel_coords=coords.astype(np.int64),
                labels=frozenset(int(v) for v in labs),
            )
        )
    return merged


def junctions_from_labels(
    labels,
    *,
    background=None,
    min_labels: int = 3,
    connectivity: int = 2,
    merge_epsilon: float = 0.0,
    start_id: int = 1,
) -> tuple[np.ndarray, list[Junction]]:
    """
    Detect, cluster, and optionally merge junctions in a label image.

    Algorithm
    ---------
    1. Scan a padded 3×3 window around every pixel and mark pixels whose
       neighborhood contains at least ``min_labels`` distinct labels.
    2. Cluster connected groups of such pixels.
    3. Assign each cluster a subpixel centroid and the union of labels observed
       around its member pixels.
    4. Optionally merge nearby junctions when ``merge_epsilon > 0``.

    Parameters
    ----------
    labels : np.ndarray
        2-D integer label image.
    background : int, optional
        Background label to exclude from junction-label counting. If ``None``
        (default), all labels participate.
    min_labels : int, optional
        Minimum distinct label count needed to mark a junction pixel.
    connectivity : int, optional
        Connectivity used to cluster junction pixels.
    merge_epsilon : float, optional
        If greater than zero, merge junction centroids within this distance.
    start_id : int, optional
        First junction ID in the returned junction-label image.

    Returns
    -------
    junction_label_image : np.ndarray
        Integer image with 1-based junction IDs and ``0`` elsewhere.
    junctions : list[Junction]
        Public junction objects containing centroids, member pixels, and label
        sets.
    """
    mask, labels_at_pixel = junction_pixels_with_labels(
        labels, background=background, min_labels=min_labels
    )
    label_image, junctions = cluster_junctions_with_labels(
        mask, labels_at_pixel, connectivity=connectivity, start_id=start_id
    )
    if merge_epsilon > 0:
        junctions = merge_close_junctions(junctions, epsilon=merge_epsilon, start_id=start_id)
        label_image = np.zeros_like(label_image)
        for junction in junctions:
            label_image[tuple(junction.pixel_coords.T)] = junction.id
    return label_image, junctions
