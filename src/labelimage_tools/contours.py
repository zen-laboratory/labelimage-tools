from __future__ import annotations

import numpy as np

from ._bbox import label_slices
from ._optional import optional_import
from .validation import validate_label_image


def _measure():
    return optional_import(
        "skimage.measure",
        extra="plot",
        feature="Contour extraction",
        package_name="scikit-image",
    )


def ordered_contour_from_mask(mask) -> np.ndarray:
    """
    Extract the longest ordered contour from a boolean mask.

    Parameters
    ----------
    mask : array-like
        Boolean or boolean-like 2-D mask for one object.

    Returns
    -------
    np.ndarray
        ``(n_points, 2)`` floating-point contour coordinates in image order
        ``(y, x)``. If no contour is found, an empty ``(0, 2)`` array is
        returned.

    Notes
    -----
    This helper wraps :func:`skimage.measure.find_contours` and selects the
    longest contour, which is usually the exterior boundary for a single
    connected label mask.
    """
    mask = np.asarray(mask, dtype=bool)
    contours = _measure().find_contours(mask.astype(float), 0.5)
    if not contours:
        return np.empty((0, 2), dtype=float)
    return max(contours, key=len).astype(float)


def ordered_contours_from_labels(labels, *, background=0) -> dict[int, np.ndarray]:
    """
    Extract one ordered contour for each non-background label.

    Parameters
    ----------
    labels : np.ndarray
        2-D integer label image.
    background : int, optional
        Label value to exclude. Default is ``0``.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping ``label -> contour``. Each contour is an ``(n_points, 2)`` array
        in ``(y, x)`` order.

    Notes
    -----
    Labels do not need to be consecutive. Each label is cropped to its local
    bounding box before being converted to a boolean mask and passed to
    :func:`ordered_contour_from_mask`.
    """
    labels = validate_label_image(labels, background=background)
    contours = {}

    # find objects (slices croping single label to its bounding box)
    slices = label_slices(labels, background=background, include_background=False, padding=1)
    
    # extract contours for each label
    for label, slc in slices.items():
        submask = labels[slc] == label
        contour = ordered_contour_from_mask(submask)
        if contour.size:
            contour = contour.copy()
            contour[:, 0] += slc[0].start
            contour[:, 1] += slc[1].start
        contours[label] = contour
    return contours
