from __future__ import annotations

import numpy as np


def validate_label_image(labels, *, background=0) -> np.ndarray:
    """
    Validate and return a 2-D integer label image as a NumPy array.

    Parameters
    ----------
    labels : array-like
        Candidate labeled image. The array must be two-dimensional and contain
        integer label values. Floating arrays are accepted only when every finite
        value is exactly integer-like, in which case they are cast to ``int64``.
    background : int, optional
        Label value used as background. The value is validated for sanity but the
        image is not required to contain it. Default is ``0``.

    Returns
    -------
    np.ndarray
        The validated label image. Integer inputs are returned as arrays without
        relabeling; integer-like floating inputs are returned as ``int64``.

    Raises
    ------
    ValueError
        If the input is not 2-D, contains non-integer values, or uses a non-finite
        background label.

    Notes
    -----
    Labels do not need to be consecutive. Values such as ``0, 5, 10`` are valid
    and are preserved.
    """
    array = np.asarray(labels)
    if array.ndim != 2:
        raise ValueError(f"label image must be 2-D, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        if np.issubdtype(array.dtype, np.floating) and np.all(np.isfinite(array)):
            rounded = np.rint(array)
            if np.allclose(array, rounded):
                array = rounded.astype(np.int64)
            else:
                raise ValueError("label image values must be integers")
        else:
            raise ValueError("label image values must be integers")
    
    validate_label_value(background, name="background")

    return array


def unique_labels(labels, *, background=0, include_background: bool = False) -> np.ndarray:
    """
    Return sorted unique labels from a 2-D label image.

    Parameters
    ----------
    labels : array-like
        Candidate labeled image accepted by :func:`validate_label_image`.
    background : int, optional
        Label value to treat as background. Default is ``0``.
    include_background : bool, optional
        If ``False`` (default), remove ``background`` from the returned values.
        If ``True``, include it when present.

    Returns
    -------
    np.ndarray
        Sorted unique label values. The original integer label values are
        preserved; no consecutiveness is assumed.
    """
    array = validate_label_image(labels, background=background)
    values = np.unique(array)
    if not include_background:
        values = values[values != background]
    return values

def validate_label_value(value, *, name: str = "label") -> int:
    """Validate and normalize a scalar label value.

    Integer scalars are accepted directly. Floating scalars are accepted only
    if finite and exactly integer-like, in which case they are rounded and
    returned as ``int``.

    Parameters
    ----------
    value : scalar
        Candidate label value.
    name : str, optional
        Name of the value for error messages. Default is ``"label"``.

    Returns
    -------
    int
        The validated label value.
    """
    array = np.asarray(value)

    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar label value")

    if np.issubdtype(array.dtype, np.integer):
        return int(array)

    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array):
            raise ValueError(f"{name} must be finite")

        rounded = np.rint(array)
        if np.allclose(array, rounded):
            return int(rounded)

    raise ValueError(f"{name} must be integer-like")

def validate_label_mapping(mapping, *, name: str = "mapping") -> dict[int, int]:
    """Validate and normalize a label replacement mapping."""
    out: dict[int, int] = {}

    for old, new in mapping.items():
        old_i = validate_label_value(old, name=f"{name} key")
        new_i = validate_label_value(new, name=f"{name}[{old_i}]")
        out[old_i] = new_i

    return out