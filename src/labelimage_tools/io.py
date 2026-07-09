from __future__ import annotations

import json
from os import PathLike
from pathlib import Path

import numpy as np

from ._graph_io import (
    LabelGraphData,
    _graph_arrays,
    _graph_data_from_json_dict,
    _graph_from_arrays,
    _graph_from_networkx,
    _graph_to_networkx,
    _infer_graph_format,
    _json_default,
    _json_dict_from_graph_data,
)
from ._optional import optional_import
from .adjacency import graph_from_labels


def _image_backend(path, backend: str) -> str:
    backend = backend.lower()
    if backend != "auto":
        if backend not in {"tifffile", "pillow"}:
            raise ValueError("backend must be one of 'auto', 'tifffile', or 'pillow'")
        return backend
    suffix = Path(path).suffix.lower()
    return "tifffile" if suffix in {".tif", ".tiff"} else "pillow"


def _tifffile():
    return optional_import(
        "tifffile",
        extra="tiff",
        feature="TIFF image I/O",
        package_name="tifffile",
    )


def _pillow_image():
    return optional_import(
        "PIL.Image",
        extra="pillow",
        feature="Pillow-backed image I/O",
        package_name="pillow",
    )


def load_img(path: str | PathLike, *, backend: str = "auto") -> np.ndarray:
    """
    Load a labeled image from disk as a NumPy array.

    Parameters
    ----------
    path : str or os.PathLike
        Path to an image file.
    backend : {"auto", "tifffile", "pillow"}, optional
        Image I/O backend. ``"auto"`` uses ``tifffile`` for ``.tif/.tiff``
        files and Pillow for other suffixes.

    Returns
    -------
    np.ndarray
        Image contents as stored in the file. Integer label values are preserved;
        no normalization, rescaling, or relabeling is applied.

    Notes
    -----
    TIFF paths use ``tifffile`` by default because that is the recommended
    backend for scientific label images. Other suffixes use Pillow by default.
    The returned array is intentionally not validated or relabeled.
    """
    backend = _image_backend(path, backend)
    if backend == "tifffile":
        return np.asarray(_tifffile().imread(path))
    image_module = _pillow_image()
    with image_module.open(path) as image:
        return np.asarray(image)


def load_label_image(path: str | PathLike, *, backend: str = "auto") -> np.ndarray:
    """
    Load a labeled image from disk.

    This is a clearer public alias for :func:`load_img`. It has the same
    behavior: preserve integer labels and avoid unnecessary image normalization.
    """
    return load_img(path, backend=backend)


def save_img(path: str | PathLike, image, *, backend: str = "auto", **kwargs) -> None:
    """
    Save an array image using ``tifffile`` or Pillow.

    ``backend="auto"`` uses ``tifffile`` for ``.tif/.tiff`` files and Pillow
    for other suffixes.
    """
    backend = _image_backend(path, backend)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if backend == "tifffile":
        _tifffile().imwrite(path, np.asarray(image), **kwargs)
        return
    image_module = _pillow_image()
    image_module.fromarray(np.asarray(image)).save(path, **kwargs)


def save_label_image(path: str | PathLike, labels, *, backend: str = "auto", **kwargs) -> None:
    """Save a label image while preserving integer label values."""
    save_img(path, labels, backend=backend, **kwargs)


def save_label_graph(
    path,
    neighbors,
    *,
    contacts=None,
    centroids=None,
    pixel_counts=None,
    format="auto",
    **metadata,
) -> None:
    """Save label adjacency graph data to NPZ, JSON, GraphML, or GEXF."""
    fmt = _infer_graph_format(path, format)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npz":
        nodes, edges, contact_values, centroid_values, pixel_count_values = _graph_arrays(
            neighbors,
            contacts=contacts,
            centroids=centroids,
            pixel_counts=pixel_counts,
        )
        arrays = {
            "nodes": nodes,
            "edges": edges,
            "metadata": np.asarray(json.dumps(metadata, default=_json_default)),
        }
        if contact_values is not None:
            arrays["contacts"] = contact_values
        if centroid_values is not None:
            arrays["centroids"] = centroid_values
        if pixel_count_values is not None:
            arrays["pixel_counts"] = pixel_count_values
        np.savez(path, **arrays)
        return
    if fmt == "json":
        with path.open("w", encoding="utf-8") as fh:
            json.dump(
                _json_dict_from_graph_data(neighbors, contacts, centroids, pixel_counts, metadata),
                fh,
                indent=2,
                default=_json_default,
            )
        return
    graph = _graph_to_networkx(neighbors, contacts, centroids, pixel_counts, metadata)
    if fmt == "graphml":
        nx = optional_import(
            "networkx",
            extra="graph-standard",
            feature="GraphML/GEXF graph I/O",
            package_name="networkx",
        )
        nx.write_graphml(graph, path)
        return
    if fmt == "gexf":
        nx = optional_import(
            "networkx",
            extra="graph-standard",
            feature="GraphML/GEXF graph I/O",
            package_name="networkx",
        )
        nx.write_gexf(graph, path)
        return
    raise AssertionError("unreachable graph format branch")



def load_label_graph(path, *, format="auto") -> LabelGraphData:
    """Load label adjacency graph data."""
    fmt = _infer_graph_format(path, format)
    path = Path(path)
    if fmt == "npz":
        with np.load(path, allow_pickle=False) as data:
            nodes = data["nodes"]
            edges = data["edges"]
            contacts = data["contacts"] if "contacts" in data.files else None
            centroids = data["centroids"] if "centroids" in data.files else None
            pixel_counts = data["pixel_counts"] if "pixel_counts" in data.files else None
            metadata = json.loads(str(data["metadata"].item())) if "metadata" in data.files else {}
        neighbors, contact_map, centroid_map, pixel_count_map = _graph_from_arrays(
            nodes,
            edges,
            contacts=contacts,
            centroids=centroids,
            pixel_counts=pixel_counts,
        )
        return LabelGraphData(neighbors, contact_map, centroid_map, pixel_count_map, metadata)
    if fmt == "json":
        with path.open(encoding="utf-8") as fh:
            return _graph_data_from_json_dict(json.load(fh))
    if fmt == "graphml":
        nx = optional_import(
            "networkx",
            extra="graph-standard",
            feature="GraphML/GEXF graph I/O",
            package_name="networkx",
        )
        return _graph_from_networkx(nx.read_graphml(path))
    if fmt == "gexf":
        nx = optional_import(
            "networkx",
            extra="graph-standard",
            feature="GraphML/GEXF graph I/O",
            package_name="networkx",
        )
        return _graph_from_networkx(nx.read_gexf(path))
    raise AssertionError("unreachable graph format branch")


def save_label_graph_from_labels(
    path,
    labels,
    *,
    background=0,
    eight=True,
    diag_weight=None,
    allow_background_contacts=False,
    include_centroids=True,
    include_pixel_counts=True,
    format="auto",
    **metadata,
) -> None:
    """Build and save label graph data directly from a label image."""
    neighbors, contacts, centroids, pixel_counts = graph_from_labels(
        labels,
        background=background,
        eight=eight,
        diag_weight=diag_weight,
        allow_background_contacts=allow_background_contacts,
        include_centroids=include_centroids,
        include_pixel_counts=include_pixel_counts,
    )
    construction_metadata = {
        "background": background,
        "eight": eight,
        "diag_weight": diag_weight,
        "allow_background_contacts": allow_background_contacts,
        "include_centroids": include_centroids,
        "include_pixel_counts": include_pixel_counts,
    }
    construction_metadata.update(metadata)
    save_label_graph(
        path,
        neighbors,
        contacts=contacts,
        centroids=centroids,
        pixel_counts=pixel_counts,
        format=format,
        **construction_metadata,
    )
