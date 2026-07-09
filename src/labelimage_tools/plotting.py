from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ._optional import optional_import
from .adjacency import adjacency_from_labels, get_centroids
from .coloring import show_map_with_colors
from .contours import ordered_contours_from_labels
from .junctions import Junction, junctions_from_labels
from .typing import Cont, Neig


def _matplotlib_pyplot():
    return optional_import(
        "matplotlib.pyplot",
        extra="plot",
        feature="Plotting",
        package_name="matplotlib",
    )


def _line_collection():
    collections = optional_import(
        "matplotlib.collections",
        extra="plot",
        feature="Plotting",
        package_name="matplotlib",
    )
    return collections.LineCollection


def _find_boundaries():
    segmentation = optional_import(
        "skimage.segmentation",
        extra="plot",
        feature="Boundary plotting",
        package_name="scikit-image",
    )
    return segmentation.find_boundaries


def _fig_ax(ax=None):
    if ax is None:
        plt = _matplotlib_pyplot()
        return plt.subplots()
    return ax.figure, ax


def draw_graph(
    im,
    neighbors: Neig,
    contacts: Cont | None = None,
    ax=None,
    show_labels: bool = True,
    show_centroids: bool = False,
    centroids: Mapping[int, np.ndarray] | None = None,
    autoadjust_ax: bool = False,
    lw_scaling: tuple[str, Any] = ("sqrt", 0.5),
    line_args=None,
    text_args=None,
    dot_args=None,
):
    """
    Draw an adjacency graph between label centroids.

    Parameters
    ----------
    im : np.ndarray or None
        Label image used to compute centroids. If ``None``, ``centroids`` must be
        supplied explicitly.
    neighbors : dict
        Adjacency mapping where keys are labels and values are neighboring
        labels.
    contacts : dict, optional
        Contact weights aligned with ``neighbors``. If omitted, all graph edges
        are drawn with unit weight.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into. If ``None``, a new figure and axis are created.
    show_labels : bool, optional
        Whether to draw label IDs at centroids. Default is ``True``.
    show_centroids : bool, optional
        Whether to mark centroids with dots. Default is ``False``.
    centroids : Mapping[int, np.ndarray], optional
        Precomputed centroid coordinates in ``(y, x)`` order. Used when ``im`` is
        ``None``.
    autoadjust_ax : bool, optional
        If ``True``, autoscale the axis, invert the y-axis, and set equal aspect.
        Leave ``False`` when drawing over an image to preserve image limits.
    lw_scaling : tuple[str, Any], optional
        Line-width scaling strategy. The first element can be ``"sqrt"`` or
        ``"linear"``; the second is a multiplicative factor.
    line_args, text_args, dot_args : dict, optional
        Additional matplotlib styling arguments for edges, labels, and centroid
        markers.

    Returns
    -------
    line_segments : matplotlib.collections.LineCollection
        Collection containing the drawn graph edges.
    ax : matplotlib.axes.Axes
        Axis containing the drawing.

    Notes
    -----
    This is adapted from ``segmentation_processing.graph.draw_graph``. Image
    coordinates are converted to matplotlib display coordinates by plotting
    ``x`` horizontally and ``y`` vertically.
    """
    line_args = {"color": "w", **(line_args or {})}
    text_args = {"color": "k", "ha": "center", "va": "center", **(text_args or {})}
    dot_args = {"color": "white", "markersize": 10, "marker": "o", **(dot_args or {})}
    if im is not None:
        centroid_dict = get_centroids(im)
    elif centroids is not None:
        centroid_dict = centroids
    else:
        raise ValueError("Either im or centroids must be provided")
    _, ax = _fig_ax(ax)
    kind, factor = lw_scaling
    if kind == "sqrt":
        def scaler(value): # type: ignore (mypy doesn't properly resolve the if branches)
            return np.sqrt(value) * factor
    elif kind == "linear":
        def scaler(value):
            return value * factor
    else:
        raise ValueError(f"unknown line width scaling kind: {kind}")

    lines = []
    widths = []
    for label, nbrs in neighbors.items():
        cy, cx = centroid_dict[int(label)]
        weights = contacts[int(label)] if contacts is not None else np.ones_like(nbrs)
        for nbr, weight in zip(nbrs, weights, strict=True):
            ny, nx = centroid_dict[int(nbr)]
            lines.append([(cx, cy), (nx, ny)])
            widths.append(scaler(weight))
        if show_labels:
            ax.text(cx, cy, str(label), **text_args)
        if show_centroids:
            ax.plot(cx, cy, **dot_args)
    collection = _line_collection()(lines, linewidths=widths, **line_args)
    ax.add_collection(collection)
    if autoadjust_ax:
        ax.autoscale()
        ax.set_ylim(ax.get_ylim()[::-1]) # type: ignore
        ax.set_aspect("equal")
    return collection, ax


def label_map(im, ax=None, background=0, **text_args):
    """
    Label each region in a labeled map at its centroid.

    Parameters
    ----------
    im : np.ndarray
        2-D integer label image.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into. If ``None``, a new figure and axis are created.
    background : int, optional
        Background label to exclude. Default is ``0``.
    **text_args
        Extra keyword arguments passed to ``Axes.text``.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the text annotations.
    """
    _, ax = _fig_ax(ax)
    text_args = {"color": "k", "ha": "center", "va": "center", **text_args}
    for label, (cy, cx) in get_centroids(im, background=background).items():
        ax.text(cx, cy, str(label), **text_args)
    return ax


def plot_label_image(
    labels,
    *,
    ax=None,
    background=0,
    use_graph_coloring: bool = True,
    K: int = 8,
    cyclic_cmap: bool = False,
    seed: int | None = None,
    title: str | None = None,
    show_colorbar: bool = False,
    interpolation: str = "nearest",
    **imshow_kwargs,
):
    """
    Plot a label image with optional graph-based coloring.

    Parameters
    ----------
    labels : np.ndarray
        2-D integer label image.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into. If ``None``, a new figure and axis are created.
    background : int, optional
        Background label. Currently used by downstream helpers and kept for
        plotting API consistency.
    use_graph_coloring : bool, optional
        If ``True`` (default), display labels through
        :func:`labelimage_tools.coloring.show_map_with_colors` so adjacent labels
        receive different colors. If ``False``, pass raw labels to ``imshow``.
    K : int, optional
        Desired graph-color palette size.
    cyclic_cmap : bool, optional
        If ``True``, assume the cmap is cyclic, so that the generated color set the
        graph-color palette does not use the start and end of the colormap, since
        they are the same color. Default is ``False``.
    seed : int, optional
        Random seed for graph-color refinement.
    title : str, optional
        Axis title.
    show_colorbar : bool, optional
        Whether to add a colorbar for the displayed image.
    interpolation : str, optional
        Interpolation passed to ``imshow``. Default is ``"nearest"``.
    **imshow_kwargs
        Additional keyword arguments passed to the image display function.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axis.

    Notes
    -----
    The function never calls ``plt.show()``, making it suitable for scripts,
    notebooks, and tests.
    """
    fig, ax = _fig_ax(ax)
    if use_graph_coloring:
        image, _, ax = show_map_with_colors(
            labels,
            ax=ax,
            K=K,
            seed=seed,
            interpolation=interpolation,
            cyclic_cmap=cyclic_cmap,
            **imshow_kwargs,
        )
    else:
        image = ax.imshow(labels, interpolation=interpolation, **imshow_kwargs)
    if show_colorbar:
        fig.colorbar(image, ax=ax)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    return fig, ax


def plot_label_boundaries(
    labels,
    *,
    ax=None,
    background=0,
    color="white",
    linewidth: float = 0.5,
    title: str | None = None,
):
    """
    Plot thick pixel boundaries between labels.

    Parameters
    ----------
    labels : np.ndarray
        2-D integer label image.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into.
    background : int, optional
        Background label. Present for API consistency; boundaries are computed
        between all neighboring label values.
    color : matplotlib color, optional
        Boundary marker color.
    linewidth : float, optional
        Marker size used for boundary pixels.
    title : str, optional
        Axis title.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axis.
    """
    fig, ax = _fig_ax(ax)
    boundaries = _find_boundaries()(np.asarray(labels), mode="thick")
    ys, xs = np.nonzero(boundaries)
    ax.plot(xs, ys, ".", color=color, markersize=linewidth)
    if title is not None:
        ax.set_title(title)
    ax.set_aspect("equal")
    return fig, ax


def plot_contours(
    labels,
    *,
    ax=None,
    background=0,
    color="white",
    linewidth: float = 0.8,
    title: str | None = None,
):
    """
    Plot ordered contours for all non-background labels.

    Parameters
    ----------
    labels : np.ndarray
        2-D integer label image.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into.
    background : int, optional
        Label value to exclude. Default is ``0``.
    color : matplotlib color, optional
        Contour line color.
    linewidth : float, optional
        Contour line width.
    title : str, optional
        Axis title.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axis.
    """
    fig, ax = _fig_ax(ax)
    for contour in ordered_contours_from_labels(labels, background=background).values():
        if len(contour):
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)
    if title is not None:
        ax.set_title(title)
    ax.set_aspect("equal")
    return fig, ax


def plot_junctions(
    labels=None,
    junctions: list[Junction] | None = None,
    *,
    junction_mask=None,
    ax=None,
    background=0,
    show_junction_ids: bool = False,
    title: str | None = None,
):
    """
    Plot junctions over an optional label image.

    Parameters
    ----------
    labels : np.ndarray, optional
        Label image to display in grayscale and/or use for junction detection.
    junctions : list[Junction], optional
        Precomputed junction objects. If omitted and ``labels`` is supplied,
        junctions are computed with :func:`junctions_from_labels`.
    junction_mask : np.ndarray, optional
        Optional mask of junction pixels to draw as small yellow points.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into.
    background : int, optional
        Background label passed to junction detection when ``junctions`` is not
        supplied.
    show_junction_ids : bool, optional
        If ``True``, draw junction IDs at their centroids.
    title : str, optional
        Axis title.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axis.
    """
    fig, ax = _fig_ax(ax)
    if labels is not None:
        ax.imshow(labels, cmap="gray", interpolation="nearest")
    if junctions is None:
        if labels is None and junction_mask is None:
            raise ValueError("provide labels, junctions, or junction_mask")
        if labels is not None:
            _, junctions = junctions_from_labels(labels, background=background)
        else:
            coords = np.argwhere(junction_mask) # type: ignore
            junctions = [
                Junction(1, coords.mean(axis=0), coords.astype(np.int64), frozenset())
            ] if len(coords) else []
    if junction_mask is not None:
        ys, xs = np.nonzero(junction_mask)
        ax.plot(xs, ys, ".", color="yellow", markersize=1)
    for junction in junctions:
        y, x = junction.yx
        ax.plot(x, y, "o", color="red", markersize=4)
        if show_junction_ids:
            ax.text(x, y, str(junction.id), color="white", ha="center", va="center")
    if title is not None:
        ax.set_title(title)
    ax.set_aspect("equal")
    return fig, ax


def plot_adjacency_graph(labels, *, ax=None, background=0, eight: bool = True):
    """
    Plot a graph-colored label image with its adjacency graph overlaid.

    Parameters
    ----------
    labels : np.ndarray
        2-D integer label image.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into.
    background : int, optional
        Background label excluded from adjacency. Default is ``0``.
    eight : bool, optional
        Whether adjacency should use 8-neighborhood contacts.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axis.
    """
    fig, ax = plot_label_image(labels, ax=ax, background=background)
    neighbors = adjacency_from_labels(labels, background=background, eight=eight)
    draw_graph(labels, neighbors, ax=ax)
    return fig, ax
