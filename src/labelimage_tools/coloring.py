from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np

from ._optional import optional_import
from .adjacency import adjacency_from_labels
from .typing import Adj, Node
from .validation import unique_labels


def _networkx_for_coloring():
    return optional_import(
        "networkx",
        extra="plot",
        feature="Graph coloring",
        package_name="networkx",
    )


def _matplotlib_for_coloring():
    plt = optional_import(
        "matplotlib.pyplot",
        extra="plot",
        feature="Graph-colored plotting",
        package_name="matplotlib",
    )
    colors = optional_import(
        "matplotlib.colors",
        extra="plot",
        feature="Graph-colored plotting",
        package_name="matplotlib",
    )
    return plt, colors


def dsatur_color(adj: Adj, seed: int | None = None) -> dict[Node, int]:
    """
    Properly color an adjacency graph with NetworkX's DSATUR heuristic.

    Parameters
    ----------
    adj : Mapping
        Adjacency mapping ``node -> iterable of neighboring nodes``.
    seed : int, optional
        Accepted for API symmetry with downstream refinement helpers. NetworkX's
        DSATUR strategy is deterministic for a fixed graph.

    Returns
    -------
    dict
        Mapping ``node -> color_index``. Color indices are compact integers
        starting at ``0``.

    Notes
    -----
    This is extracted from ``segmentation_processing.coloring.dsatur_color`` and
    uses NetworkX's ``saturation_largest_first`` greedy-coloring strategy.
    """
    nx = _networkx_for_coloring()
    graph = nx.Graph()
    for node, neighbors in adj.items():
        graph.add_node(node)
        for neighbor in neighbors:
            if node != neighbor:
                graph.add_edge(node, neighbor)
    return nx.algorithms.coloring.greedy_color(graph, strategy="saturation_largest_first")


def refine_to_K_colors(
    base_color: Mapping[Node, int],
    K: int,
    seed: int | None = None,
    balance: str = "proportional",
) -> dict[Node, int]:
    """
    Split DSATUR color classes to use exactly ``K`` colors.

    The input ``base_color`` is assumed to be a proper coloring. Each base color
    class is an independent set, so splitting nodes *within* a class into several
    color variants cannot introduce adjacency conflicts.

    Parameters
    ----------
    base_color : Mapping
        Initial proper coloring, usually returned by :func:`dsatur_color`.
    K : int
        Desired number of output colors. Must be at least the number of base
        color classes and no larger than the number of colored nodes if exactly
        ``K`` colors are to be used.
    seed : int, optional
        Random seed used when shuffling nodes inside each base class.
    balance : {"proportional", "even"}, optional
        Strategy for assigning variants to base classes. ``"proportional"``
        gives larger independent sets more variants. ``"even"`` distributes
        variants as evenly as possible across base classes.

    Returns
    -------
    dict
        Refined color mapping using exactly ``K`` color indices.
    """
    rng = np.random.default_rng(seed)
    classes: dict[int, list[Node]] = defaultdict(list)
    for node, color in base_color.items():
        classes[int(color)].append(node)
    c_count = len(classes)
    if K < c_count:
        raise ValueError(f"Requested K={K} < base colors {c_count}")
    sizes = np.array([len(classes[c]) for c in range(c_count)], dtype=float)
    if balance == "proportional" and sizes.sum() > 0:
        target = K * (sizes / sizes.sum())
        variants = np.floor(target).astype(int)
        variants = np.maximum(variants, 1)
        while variants.sum() < K:
            variants[int(np.argmax(target - variants))] += 1
        while variants.sum() > K:
            variants[int(np.argmax(variants))] -= 1
    else:
        q, r = divmod(K, c_count)
        variants = np.array([q + (1 if i < r else 0) for i in range(c_count)], dtype=int)

    refined = {}
    next_idx = 0
    for base in range(c_count):
        palette = list(range(next_idx, next_idx + int(variants[base])))
        next_idx += int(variants[base])
        nodes = list(classes[base])
        rng.shuffle(nodes)
        for j, node in enumerate(nodes):
            refined[node] = palette[j % len(palette)]
    if refined and len(set(refined.values())) != K:
        raise RuntimeError("could not refine coloring to exactly K colors")
    return refined


def rebalance_K_colors(
    adj: Adj,
    color: Mapping[Node, int],
    K: int,
    *,
    seed: int | None = None,
    max_rounds: int = 10,
    tolerance: float = 0.1,
    protect_singletons: bool = True,
) -> dict[Node, int]:
    """
    Heuristically balance K color classes without creating conflicts.

    Starting from a valid K-coloring, this routine repeatedly tries to move
    nodes from the largest color class into the smallest color class. A move is
    accepted only if none of the node's neighbors currently use the target color.

    Parameters
    ----------
    adj : Mapping
        Adjacency mapping.
    color : Mapping
        Initial proper coloring with values in ``0..K-1``. It is not mutated.
    K : int
        Number of colors in use.
    seed : int, optional
        Random seed for shuffling candidate nodes during each round.
    max_rounds : int, optional
        Maximum number of rebalancing passes.
    tolerance : float, optional
        Target balance tolerance, as a fraction of the ideal class size
        ``N / K``.
    protect_singletons : bool, optional
        If ``True``, never move the last node out of a color class, preserving
        use of all K colors.

    Returns
    -------
    dict
        New coloring mapping that remains conflict-free.
    """
    if not color:
        return dict(color)
    rng = np.random.default_rng(seed)
    new = dict(color)
    ideal = len(new) / float(K)
    max_diff = max(1, int(np.ceil(tolerance * ideal)))
    for _ in range(max_rounds):
        counts = np.bincount(np.fromiter(new.values(), dtype=int), minlength=K)
        smallest = int(np.argmin(counts))
        largest = int(np.argmax(counts))
        if counts[largest] - counts[smallest] <= max_diff:
            break
        candidates = [node for node, col in new.items() if col == largest]
        rng.shuffle(candidates)
        moved = False
        for node in candidates:
            if protect_singletons and counts[largest] <= 1:
                break
            if any(new.get(neighbor) == smallest for neighbor in adj.get(node, ())):
                continue
            new[node] = smallest
            counts[largest] -= 1
            counts[smallest] += 1
            moved = True
            break
        if not moved:
            break
    return new


def apply_color_lut_int(
    map_array: np.ndarray,
    color_mapping: dict[Node, int] | None = None,
    lut: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply an integer label-to-color lookup table.

    Parameters
    ----------
    map_array : np.ndarray
        2-D integer array of non-negative labels.
    color_mapping : dict, optional
        Mapping ``label -> color_index``. Required if ``lut`` is not supplied.
    lut : np.ndarray, optional
        Precomputed lookup table where ``lut[label]`` is a color index and
        unmapped labels contain ``NaN``.

    Returns
    -------
    mapped : np.ndarray
        Floating-point array with color indices and ``NaN`` where labels are not
        mapped.
    lut : np.ndarray
        Lookup table used for the mapping.

    Notes
    -----
    This preserves the fast vectorized LUT behavior from the source
    ``apply_color_lut_int`` helper, with an additional guard for labels outside a
    provided LUT.
    """
    array = np.asarray(map_array)
    if array.size == 0:
        return np.empty_like(array, dtype=np.float32), np.empty((0,), dtype=np.float32)
    if np.any(array < 0):
        raise ValueError("apply_color_lut_int expects non-negative labels")
    if lut is None:
        if color_mapping is None:
            raise ValueError("Either color_mapping or lut must be provided")
        max_label = int(max([int(array.max()), *(int(label) for label in color_mapping)]))
        lut = np.full(max_label + 1, np.nan, dtype=np.float32)
        for label, color in color_mapping.items():
            if int(label) >= 0:
                lut[int(label)] = float(color)
    out = np.full(array.shape, np.nan, dtype=np.float32)
    in_range = array <= (len(lut) - 1)
    out[in_range] = lut[array[in_range]]
    return out, lut


def color_planar_with_variety(
    adj: Adj,
    K: int = 8,
    seed: int | None = None,
    balance: str = "proportional",
    rebalance: bool = True,
) -> dict[Node, int]:
    """
    Produce a conflict-free coloring with exactly ``K`` colors when feasible.

    Internally this performs:

    1. DSATUR base coloring, usually using few colors on planar adjacency graphs.
    2. Refinement of independent sets with :func:`refine_to_K_colors` when
       ``K`` is larger than the base color count.
    3. Optional graph-aware rebalancing with :func:`rebalance_K_colors`.

    Parameters
    ----------
    adj : Mapping
        Label adjacency mapping.
    K : int, optional
        Desired number of colors.
    seed : int, optional
        Random seed for refinement and rebalancing.
    balance : {"proportional", "even"}, optional
        Refinement strategy.
    rebalance : bool, optional
        If ``True``, try to make color classes more even without introducing
        conflicts.

    Returns
    -------
    dict
        Mapping ``label -> color_index``.

    Raises
    ------
    ValueError
        If the DSATUR base coloring needs more than ``K`` colors, or if there
        are fewer labels than requested colors.
    """
    base = dsatur_color(adj, seed=seed)
    if K > len(base):
        raise ValueError(f"Requested K={K} colors for only {len(base)} labels")
    used = len(set(base.values()))
    if used == K:
        colored = dict(base)
    elif used < K:
        colored = refine_to_K_colors(base, K, seed=seed, balance=balance)
    else:
        raise ValueError(f"Base coloring used {used} colors; choose K >= {used}")
    if rebalance:
        colored = rebalance_K_colors(adj, colored, K, seed=seed)
    return colored


def show_map_with_colors(
    map_array: np.ndarray,
    ax=None,
    cmap="tab20",
    cyclic_cmap: bool = False,
    adj: Adj | None = None,
    lut: np.ndarray | None = None,
    K: int = 8,
    seed: int | None = None,
    balance: str = "proportional",
    rebalance: bool = True,
    holes_separate: bool = True,
    hole_color: Any = "0.3",
    **imshow_kwargs,
):
    """
    Display a labeled map with conflict-free graph-based coloring.

    Parameters
    ----------
    map_array : np.ndarray
        2-D integer label image.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into. If ``None``, a new figure and axis are created.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap used for color indices. Default is ``"tab20"``.
    cyclic_cmap : bool, optional
        If ``True``, normalize with an extra color step for cyclic palettes.
    adj : Mapping, optional
        Precomputed adjacency mapping. If omitted, adjacency is computed from
        ``map_array`` with background ``0`` excluded.
    lut : np.ndarray, optional
        Precomputed label-to-color lookup table. If supplied, graph coloring is
        skipped and this LUT is used directly.
    K : int, optional
        Desired palette size. For tiny images with fewer labels than ``K``, the
        displayed palette is reduced so plotting remains convenient. Isolated
        labels with no adjacency edges are still included in the color mapping.
    seed : int, optional
        Random seed for color refinement/rebalancing.
    balance : {"proportional", "even"}, optional
        Refinement strategy.
    rebalance : bool, optional
        Whether to rebalance color classes after refinement.
    holes_separate : bool, optional
        If ``True``, sentinel labels greater than ``9999`` are shown using the
        colormap's over color.
    hole_color : Any, optional
        Color used for sentinel hole labels when ``holes_separate=True``.
    **imshow_kwargs
        Extra keyword arguments passed to ``Axes.imshow``.

    Returns
    -------
    image : matplotlib.image.AxesImage
        Image artist returned by ``imshow``.
    lut : np.ndarray
        Lookup table used to map label values to color indices.
    ax : matplotlib.axes.Axes
        Axis containing the image.
    """
    plt, colors = _matplotlib_for_coloring()
    if ax is None:
        _, ax = plt.subplots()
    if lut is None:
        if adj is None:
            adj = adjacency_from_labels(map_array, background=0)
        adj = {node: list(neighbors) for node, neighbors in adj.items()}
        for label in unique_labels(map_array, background=0):
            adj.setdefault(int(label), [])
        K_effective = min(K, len(adj))
        color_mapping = (
            color_planar_with_variety(
                adj, K=K_effective, seed=seed, balance=balance, rebalance=rebalance
            )
            if K_effective > 0
            else {}
        )
    else:
        color_mapping = None
        K_effective = K
    mapped, lut = apply_color_lut_int(map_array, color_mapping, lut=lut)
    if holes_separate:
        cmap = plt.get_cmap(cmap).copy()
        cmap.set_over(hole_color)
        mapped[np.asarray(map_array) > 9999] = K + 1
    vmax = max(1, K_effective if cyclic_cmap else K_effective - 1)
    norm = colors.Normalize(vmin=0, vmax=vmax)
    image = ax.imshow(mapped, cmap=cmap, norm=norm, **imshow_kwargs)
    return image, lut, ax
