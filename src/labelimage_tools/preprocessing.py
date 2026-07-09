from __future__ import annotations

from os import PathLike

import numpy as np
from scipy import ndimage as ndi

from ._bbox import label_slices
from .io import load_img
from .validation import validate_label_image


def _structure(structure=None) -> np.ndarray:
    if structure is None:
        return np.ones((3, 3), dtype=bool)
    if isinstance(structure, int):
        return np.ones((structure, structure), dtype=bool)
    return np.asarray(structure, dtype=bool)


def erode_labels(im, structure=None, background=0) -> np.ndarray:
    """
    Erode each labeled region independently using binary erosion.

    Parameters
    ----------
    im : np.ndarray
        2-D integer array representing a labeled map.
    structure : int or np.ndarray, optional
        Structuring element used for erosion. If an integer is provided, a square
        structuring element of that size is used. If ``None``, a 3×3 square
        structuring element is used.
    background : int, optional
        Label value representing background. Default is ``0``.

    Returns
    -------
    np.ndarray
        Label image with each non-background object eroded. Pixels removed by
        erosion become ``background``. Small regions may disappear completely.

    """
    labels = validate_label_image(im, background=background)
    structure = _structure(structure)
    out = np.full(labels.shape, background, dtype=labels.dtype)
    for label, slc in label_slices(labels, background=background).items():
        # Erode only this label's object mask; eroded-away pixels stay background.
        sub = labels[slc] == label
        out[slc][ndi.binary_erosion(sub, structure=structure)] = label
    return out


def dilate_labels(im, structure=None, background=0, background_only: bool = True) -> np.ndarray:
    """
    Dilate each labeled region independently using binary dilation.

    Parameters
    ----------
    im : np.ndarray
        2-D integer array representing a labeled map.
    structure : int or np.ndarray, optional
        Structuring element used for dilation. If an integer is provided, a square
        structuring element of that size is used. If ``None``, a 3×3 square
        structuring element is used.
    background : int, optional
        Label value representing background. Default is ``0``.
    background_only : bool, optional
        If ``True`` (default), labels expand only into pixels that were
        ``background`` in the input. If ``False``, dilation may overwrite other
        labels in ascending label order, matching the behavior of the source
        implementation.

    Returns
    -------
    np.ndarray
        Dilated label image.

    Notes
    -----
    This function is the correctly spelled public version of the original
    ``dialate_labels`` helper. The old spelling is kept as an alias below.
    """
    labels = validate_label_image(im, background=background)
    structure = _structure(structure)

    # Pad each object crop enough to include the full dilation footprint.
    rowpad, colpad = structure.shape[0] // 2, structure.shape[1] // 2
    out = labels.copy()
    bg_mask = labels == background
    slices = label_slices(labels, background=background, padding=max(rowpad, colpad))
    for label, slc in slices.items():
        # Dilate the single-label mask, then decide whether expansion is allowed
        # only into background pixels or may overwrite existing labels.
        sub = labels[slc] == label
        dilated = ndi.binary_dilation(sub, structure=structure)
        if background_only:
            out[slc][dilated & bg_mask[slc]] = label
        else:
            out[slc][dilated] = label
    return out


def dialate_labels(im, structure=None, background=0, background_only: bool = True) -> np.ndarray:
    """
    Behavior-preserving misspelled alias for :func:`dilate_labels`.

    The original helper in ``segmentation_processing.img_treatment`` was named
    ``dialate_labels``. New code should use :func:`dilate_labels`, but this alias
    is kept because existing internal notebooks and scripts may still use the
    misspelled name.
    """
    # Keep the original misspelled entry point as a thin wrapper.
    return dilate_labels(
        im,
        structure=structure,
        background=background,
        background_only=background_only,
    )


def shuffle_labels(im, seed=None, background=None) -> np.ndarray:
    """
    Randomly shuffle label values in a label image.

    Parameters
    ----------
    im : np.ndarray
        2-D integer array representing a labeled map.
    seed : int, optional
        Random seed for reproducible shuffling.
    background : int, optional
        Background label to preserve. If provided, this label maps to itself and
        is not included in the random permutation. If ``None``, every label is
        eligible for shuffling.

    Returns
    -------
    np.ndarray
        Label image with the same spatial regions but permuted label values.

    Notes
    -----
    Label values are shuffled directly; labels are not compacted or made
    consecutive.
    """
    labels = validate_label_image(im, background=background if background is not None else 0)
    rng = np.random.default_rng(seed)
    values = np.unique(labels)
    to_shuffle = values if background is None else values[values != background]
    shuffled = to_shuffle.copy()
    rng.shuffle(shuffled)
    mapping = {old: new for old, new in zip(to_shuffle, shuffled, strict=True)}
    if background is not None:
        mapping[background] = background
    return np.vectorize(mapping.get, otypes=[labels.dtype])(labels)


def fill_internal_gaps_edt(
    labels,
    background=0,
    max_distance=None,
    fill_value: int = 10_000,
) -> np.ndarray:
    """
    Fill internal background holes with nearest labels using Euclidean distance.

    Parameters
    ----------
    labels : np.ndarray
        2-D integer label image.
    background : int, optional
        Background label. Default is ``0``.
    max_distance : float, optional
        If ``None``, all pixels in internal holes are filled with the nearest
        foreground label. If provided, only hole pixels whose nearest-foreground
        distance is at most ``max_distance`` are filled with real labels.
        Farther pixels receive sentinel labels.
    fill_value : int, optional
        First sentinel label assigned to far pixels when ``max_distance`` is
        provided. Each far connected hole component receives ``fill_value``,
        then ``fill_value + 1``, and so on.

    Returns
    -------
    np.ndarray
        Copy of ``labels`` with internal background gaps filled.

    Notes
    -----
    "Internal gaps" are background connected components fully enclosed by
    foreground. Background connected to the image border is not filled. 
    """
    labels = validate_label_image(labels, background=background)
    fg = labels != background
    out = labels.copy()

    # Sentinel labels must not collide with real labels.
    max_label = int(labels.max()) if labels.size else 0
    if fill_value <= max_label:
        raise ValueError("fill_value must be larger than all existing labels")

    # Internal holes are background components fully enclosed by foreground.
    filled_fg = ndi.binary_fill_holes(fg)
    holes = filled_fg & (~fg)
    if not np.any(holes):
        return labels.copy()

    # Compute nearest-foreground assignments for all background pixels once.
    distances, inds = ndi.distance_transform_edt(~fg, return_distances=True, return_indices=True)  # type: ignore (linter thinks this could return None)
    assign_all_bg = labels[tuple(inds)]

    # Work hole by hole when a distance threshold can leave sentinel regions.
    cc, n_cc = ndi.label(holes) # type: ignore (linter think we could get None returned here)
    if n_cc == 0:
        return labels.copy()
    if max_distance is None:
        # Without a threshold, every internal hole pixel receives its nearest
        # real label directly.
        out[holes] = assign_all_bg[holes]
        return out

    for idx, slc in label_slices(cc, background=0).items():
        hole_mask = cc[slc] == idx
        sub_dist = distances[slc]
        sub_assign = assign_all_bg[slc]

        # Within each hole, near pixels get real labels and far pixels get one
        # unique sentinel label for that connected hole component.
        far = hole_mask & (sub_dist > max_distance)
        close = hole_mask & ~far
        if np.any(close):
            out[slc][close] = sub_assign[close]
        if np.any(far):
            out[slc][far] = fill_value
            fill_value += 1
    return out


def skeletonize_dilate(labels, background=0) -> np.ndarray:
    """
    Produce a one-pixel exterior border around each label.

    The border pixels are assigned the label of the object they border. To get a
    binary skeletonized image, use ``out != background`` on the returned array.

    Parameters
    ----------
    labels : np.ndarray
        2-D integer label image.
    background : int, optional
        Background label. Default is ``0``.

    Returns
    -------
    np.ndarray
        Label image containing exterior border pixels for each non-background
        label.
    """
    labels = validate_label_image(labels, background=background)
    out = np.full(labels.shape, background, dtype=labels.dtype)

    # Cross-shaped structure gives a 4-connected exterior border.
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    for label, slc in label_slices(labels, background=background, padding=1).items():
        # Dilate the object and keep only the newly reached pixels.
        sub = labels[slc] == label
        border = ndi.binary_dilation(sub, structure=struct) & (~sub)
        out[slc][border] = label
    return out


def skeletonize_erode(labels, background=0) -> np.ndarray:
    """
    Produce a one-pixel interior border for each label.

    The border pixels are assigned the label of the object they belong to. To get
    a binary skeletonized image, use ``out != background`` on the returned array.

    Parameters
    ----------
    labels : np.ndarray
        2-D integer label image.
    background : int, optional
        Background label. Default is ``0``.

    Returns
    -------
    np.ndarray
        Label image containing interior border pixels for each non-background
        label.
    """
    labels = validate_label_image(labels, background=background)
    out = np.full(labels.shape, background, dtype=labels.dtype)

    # Cross-shaped structure gives a 4-connected interior border.
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    for label, slc in label_slices(labels, background=background).items():
        # Erode the object and keep the pixels removed by erosion.
        sub = labels[slc] == label
        border = (~ndi.binary_erosion(sub, structure=struct)) & sub
        out[slc][border] = label
    return out


def skeletonize_labels(labels, background=0, kind: str = "interior") -> np.ndarray:
    """
    Produce a one-pixel skeleton of each label.

    This is a convenience wrapper around :func:`skeletonize_erode` and
    :func:`skeletonize_dilate`.

    Parameters
    ----------
    labels : np.ndarray
        2-D integer label image.
    background : int, optional
        Background label. Default is ``0``.
    kind : {"interior", "exterior"}, optional
        ``"interior"`` returns the eroded/interior skeleton. ``"exterior"``
        returns the dilated/exterior skeleton.

    Returns
    -------
    np.ndarray
        Skeletonized label image.
    """
    if kind == "interior":
        return skeletonize_erode(labels, background=background)
    if kind == "exterior":
        return skeletonize_dilate(labels, background=background)
    raise ValueError("kind must be 'interior' or 'exterior'")


def find_non_self_connected_labels(
    im,
    background=0,
    connectivity: int = 1,
) -> dict[int, np.ndarray]:
    """
    Find labels whose pixels form more than one connected component.

    Parameters
    ----------
    im : np.ndarray
        2-D integer label image.
    background : int, optional
        Label value to ignore. Default is ``0``.
    connectivity : int, optional
        Connectivity passed to :func:`scipy.ndimage.generate_binary_structure`.
        In 2-D, ``1`` is 4-connectivity and ``2`` is 8-connectivity.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping ``label -> component centroids`` for labels with multiple
        disconnected blobs. Centroids are in image coordinates ``(y, x)``.

    """
    labels = validate_label_image(im, background=background)

    # Choose whether components are connected through edges only or also corners.
    structure = ndi.generate_binary_structure(labels.ndim, connectivity)
    bad = {}
    for label, slc in label_slices(labels, background=background, padding=1).items():
        sub = labels[slc] == label

        # Label connected components within this single cell/object.
        cc_labels, n_cc = ndi.label(sub, structure=structure)  # type: ignore (linter thinks this could return None)
        if n_cc > 1:
            # Return component centroids in global image coordinates.
            centers = ndi.center_of_mass(sub, cc_labels, index=range(1, n_cc + 1))
            bad[int(label)] = np.asarray(
                [(cy + slc[0].start, cx + slc[1].start) for cy, cx in centers], dtype=float
            )
    return bad


def remove_non_self_connected_bits(im, background=0, connectivity: int = 1) -> np.ndarray:
    """
    Remove disconnected fragments from labels, keep the largest component.

    For each non-background label, connected components are computed within that
    label. If a label has more than one component, only the largest component is
    kept and all smaller components are relabeled as ``background``.

    Parameters
    ----------
    im : np.ndarray
        2-D integer label image.
    background : int, optional
        Background label. Default is ``0``.
    connectivity : int, optional
        Connectivity for component labeling. In 2-D, ``1`` is 4-connectivity and
        ``2`` is 8-connectivity.

    Returns
    -------
    np.ndarray
        Cleaned label image.
    """
    labels = validate_label_image(im, background=background)

    # Choose whether components are connected through edges only or also corners.
    structure = ndi.generate_binary_structure(labels.ndim, connectivity)
    cleaned = labels.copy()
    for label, slc in label_slices(labels, background=background, padding=1).items():
        sub = labels[slc] == label

        # Label connected components within this single cell/object.
        cc_labels, n_cc = ndi.label(sub, structure=structure)  # type: ignore (linter thinks this could return None)
        if n_cc > 1:
            # Keep only the largest component; smaller fragments become background.
            sizes = ndi.sum(sub, cc_labels, index=range(1, n_cc + 1))
            largest = int(np.argmax(sizes)) + 1
            cleaned[slc][(cc_labels != largest) & (cc_labels != 0)] = background
    return cleaned


def crop_to_foreground_bbox(
    im,
    background=0,
    padding: int = 20,
) -> tuple[np.ndarray, tuple[slice, slice]]:
    """
    Crop a label image to the foreground bounding box plus padding.

    Parameters
    ----------
    im : np.ndarray
        2-D integer label image.
    background : int, optional
        Background label. Default is ``0``.
    padding : int, optional
        Number of pixels to include on each side of the foreground bounding box.
        The image is never expanded, so actual padding may be smaller near image
        borders.

    Returns
    -------
    cropped : np.ndarray
        View/copy-like slice of the input containing all non-background pixels
        plus available padding.
    slices : tuple[slice, slice]
        Row and column slices used to produce ``cropped``.
    """
    labels = validate_label_image(im, background=background)
    rows = np.any(labels != background, axis=1)
    cols = np.any(labels != background, axis=0)
    if not np.any(rows) or not np.any(cols):
        return labels, (slice(0, labels.shape[0]), slice(0, labels.shape[1]))
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    srow = slice(max(0, rmin - padding), min(labels.shape[0], rmax + padding + 1))
    scol = slice(max(0, cmin - padding), min(labels.shape[1], cmax + padding + 1))
    return labels[srow, scol], (srow, scol)


def load_image_pipeline(
    path: str | PathLike,
    seed=None,
    background=0,
    backend: str = "auto",
    connectivity: int = 1,
    crop_to_foreground: bool = True,
    remove_small_bits: bool = True,
    fill_holes: bool = True,
    dilate_borders: bool = False,
    shuffle: bool = False,
) -> np.ndarray:
    """
    Load a label image and run the standard cleaning pipeline.

    The pipeline mirrors the source ``img_treatment.load_image_pipeline`` helper:
    load the image, optionally crop to foreground, remove disconnected label
    fragments, fill internal holes, optionally dilate borders, and optionally
    shuffle label values.

    Parameters
    ----------
    path : str or os.PathLike
        Path to a labeled image file.
    seed : int, optional
        Random seed used when ``shuffle=True``.
    background : int, optional
        Background label. Default is ``0``.
    backend : {"auto", "tifffile", "pillow"}, optional
        Image I/O backend passed to :func:`labelimage_tools.load_img`.
    connectivity : int, optional
        Connectivity used for disconnected-fragment cleanup.
    crop_to_foreground : bool, optional
        If ``True``, crop to the non-background bounding box before other
        operations.
    remove_small_bits : bool, optional
        If ``True``, remove disconnected fragments of each label.
    fill_holes : bool, optional
        If ``True``, fill internal background holes using
        :func:`fill_internal_gaps_edt`.
    dilate_borders : bool, optional
        If ``True``, dilate labels into background and fill any new internal gaps.
    shuffle : bool, optional
        If ``True``, randomly permute label values while preserving background.

    Returns
    -------
    np.ndarray
        Processed label image.
    """
    im = load_img(path, backend=backend)
    if crop_to_foreground:
        im, _ = crop_to_foreground_bbox(im, background=background, padding=5)
    if remove_small_bits:
        im = remove_non_self_connected_bits(im, background=background, connectivity=connectivity)
    if fill_holes:
        im = fill_internal_gaps_edt(im, background=background, max_distance=3)
    if dilate_borders:
        im = dilate_labels(im, background=background)
        im = fill_internal_gaps_edt(im, background=background, max_distance=3)
    if shuffle:
        im = shuffle_labels(im, seed=seed, background=background)
    return im
