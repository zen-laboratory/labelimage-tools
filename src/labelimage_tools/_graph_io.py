from pathlib import Path
from collections import namedtuple
import json

import numpy as np
import networkx as nx

from .typing import Cont, Neig


LabelGraphData = namedtuple(
    "LabelGraphData",
    ["neighbors", "contacts", "centroids", "pixel_counts", "metadata"],
)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _infer_graph_format(path, format: str) -> str:
    fmt = format.lower()
    if fmt != "auto":
        if fmt not in {"npz", "json", "graphml", "gexf"}:
            raise ValueError("format must be one of 'auto', 'npz', 'json', 'graphml', or 'gexf'")
        return fmt
    suffix = Path(path).suffix.lower()
    if suffix == ".npz":
        return "npz"
    if suffix == ".json":
        return "json"
    if suffix == ".graphml":
        return "graphml"
    if suffix == ".gexf":
        return "gexf"
    raise ValueError(
        "could not infer graph format from suffix; use .npz, .json, .graphml, or .gexf"
    )


def _graph_arrays(
    neighbors: Neig,
    *,
    contacts: Cont | None = None,
    centroids: dict[int, np.ndarray] | None = None,
    pixel_counts: dict[int, int] | None = None,
):
    node_set = {int(label) for label in neighbors}
    for nbrs in neighbors.values():
        node_set.update(int(nbr) for nbr in np.asarray(nbrs).ravel())
    if centroids is not None:
        node_set.update(int(label) for label in centroids)
    if pixel_counts is not None:
        node_set.update(int(label) for label in pixel_counts)
    nodes = np.asarray(sorted(node_set), dtype=np.int64)

    edge_contacts: dict[tuple[int, int], float] = {}
    edge_set: set[tuple[int, int]] = set()
    for label, nbrs in neighbors.items():
        label = int(label)
        weights = contacts.get(label) if contacts is not None else None
        if weights is None:
            weights = [None] * len(nbrs)
        for nbr, weight in zip(np.asarray(nbrs).ravel(), weights, strict=True):
            edge = tuple(sorted((label, int(nbr))))
            if edge[0] == edge[1]:
                continue
            edge_set.add(edge)
            if contacts is not None and weight is not None:
                edge_contacts.setdefault(edge, float(weight))

    edges = np.asarray(sorted(edge_set), dtype=np.int64).reshape(-1, 2)
    contact_values = (
        np.asarray([edge_contacts[tuple(edge)] for edge in edges], dtype=float)
        if contacts is not None
        else None
    )
    centroid_values = (
        np.asarray(
            [np.asarray(centroids.get(int(node), [np.nan, np.nan]), dtype=float) for node in nodes],
            dtype=float,
        )
        if centroids is not None
        else None
    )
    pixel_count_values = (
        np.asarray([int(pixel_counts.get(int(node), 0)) for node in nodes], dtype=np.int64)
        if pixel_counts is not None
        else None
    )
    return nodes, edges, contact_values, centroid_values, pixel_count_values


def _graph_from_arrays(
    nodes,
    edges,
    *,
    contacts=None,
    centroids=None,
    pixel_counts=None,
):
    neighbors_lists: dict[int, list[int]] = {int(node): [] for node in np.asarray(nodes).ravel()}
    contact_lists: dict[int, list[float]] | None = (
        {int(node): [] for node in np.asarray(nodes).ravel()} if contacts is not None else None
    )
    for idx, edge in enumerate(np.asarray(edges, dtype=np.int64).reshape(-1, 2)):
        a, b = int(edge[0]), int(edge[1])
        neighbors_lists.setdefault(a, []).append(b)
        neighbors_lists.setdefault(b, []).append(a)
        if contact_lists is not None:
            weight = float(np.asarray(contacts, dtype=float)[idx])
            contact_lists.setdefault(a, []).append(weight)
            contact_lists.setdefault(b, []).append(weight)

    neighbors = {
        label: np.asarray(values, dtype=np.int64)
        for label, values in neighbors_lists.items()
    }
    contact_map = (
        {label: np.asarray(values, dtype=float) for label, values in contact_lists.items()}
        if contact_lists is not None
        else None
    )
    centroid_map = (
        {
            int(node): np.asarray(value, dtype=float)
            for node, value in zip(nodes, centroids, strict=True)
        }
        if centroids is not None
        else None
    )
    pixel_count_map = (
        {int(node): int(value) for node, value in zip(nodes, pixel_counts, strict=True)}
        if pixel_counts is not None
        else None
    )
    return neighbors, contact_map, centroid_map, pixel_count_map


def _json_dict_from_graph_data(neighbors, contacts, centroids, pixel_counts, metadata):
    nodes, edges, contact_values, _, _ = _graph_arrays(
        neighbors,
        contacts=contacts,
        centroids=centroids,
        pixel_counts=pixel_counts,
    )
    node_items = []
    for node in nodes:
        item = {"id": int(node)}
        if centroids is not None and int(node) in centroids:
            item["centroid"] = np.asarray(centroids[int(node)], dtype=float).tolist()
        if pixel_counts is not None and int(node) in pixel_counts:
            item["pixel_count"] = int(pixel_counts[int(node)])
        node_items.append(item)
    edge_items = []
    for idx, edge in enumerate(edges):
        item = {"source": int(edge[0]), "target": int(edge[1])}
        if contact_values is not None:
            contact = float(contact_values[idx])
            item["contact"] = contact
            item["weight"] = contact
        edge_items.append(item)
    return {"nodes": node_items, "edges": edge_items, "metadata": dict(metadata)}


def _graph_data_from_json_dict(data):
    nodes = np.asarray([int(node["id"]) for node in data.get("nodes", [])], dtype=np.int64)
    edges = np.asarray(
        [[int(edge["source"]), int(edge["target"])] for edge in data.get("edges", [])],
        dtype=np.int64,
    ).reshape(-1, 2)
    has_contacts = any("contact" in edge for edge in data.get("edges", []))
    contacts = (
        np.asarray([float(edge.get("contact", edge.get("weight", 1.0))) for edge in data["edges"]])
        if has_contacts
        else None
    )
    has_centroids = any("centroid" in node for node in data.get("nodes", []))
    centroids = (
        np.asarray([node.get("centroid", [np.nan, np.nan]) for node in data["nodes"]], dtype=float)
        if has_centroids
        else None
    )
    has_pixel_counts = any("pixel_count" in node for node in data.get("nodes", []))
    pixel_counts = (
        np.asarray([int(node.get("pixel_count", 0)) for node in data["nodes"]], dtype=np.int64)
        if has_pixel_counts
        else None
    )
    neighbors, contact_map, centroid_map, pixel_count_map = _graph_from_arrays(
        nodes,
        edges,
        contacts=contacts,
        centroids=centroids,
        pixel_counts=pixel_counts,
    )
    return LabelGraphData(
        neighbors,
        contact_map,
        centroid_map,
        pixel_count_map,
        dict(data.get("metadata", {})),
    )


def _graph_to_networkx(neighbors, contacts=None, centroids=None, pixel_counts=None, metadata=None):
    graph = nx.Graph()
    nodes, edges, contact_values, _, _ = _graph_arrays(
        neighbors,
        contacts=contacts,
        centroids=centroids,
        pixel_counts=pixel_counts,
    )
    for node in nodes:
        attrs = {}
        if centroids is not None and int(node) in centroids:
            cy, cx = np.asarray(centroids[int(node)], dtype=float)
            attrs.update({"centroid_y": float(cy), "centroid_x": float(cx)})
        if pixel_counts is not None and int(node) in pixel_counts:
            attrs["pixel_count"] = int(pixel_counts[int(node)])
        graph.add_node(str(int(node)), **attrs)
    for idx, edge in enumerate(edges):
        attrs = {}
        if contact_values is not None:
            contact = float(contact_values[idx])
            attrs.update({"contact": contact, "weight": contact})
        graph.add_edge(str(int(edge[0])), str(int(edge[1])), **attrs)
    if metadata:
        graph.graph["metadata"] = json.dumps(metadata, default=_json_default)
    return graph


def _graph_from_networkx(graph):
    nodes = np.asarray([int(node) for node in graph.nodes], dtype=np.int64)
    edges = np.asarray([[int(a), int(b)] for a, b in graph.edges], dtype=np.int64).reshape(-1, 2)
    has_contacts = any(
        "contact" in data or "weight" in data
        for _, _, data in graph.edges(data=True)
    )
    contacts = (
        np.asarray(
            [
                float(data.get("contact", data.get("weight", 1.0)))
                for _, _, data in graph.edges(data=True)
            ],
            dtype=float,
        )
        if has_contacts
        else None
    )
    has_centroids = any(
        "centroid_y" in data and "centroid_x" in data
        for _, data in graph.nodes(data=True)
    )
    centroids = (
        np.asarray(
            [
                [float(data.get("centroid_y", np.nan)), float(data.get("centroid_x", np.nan))]
                for _, data in graph.nodes(data=True)
            ],
            dtype=float,
        )
        if has_centroids
        else None
    )
    has_pixel_counts = any("pixel_count" in data for _, data in graph.nodes(data=True))
    pixel_counts = (
        np.asarray(
            [int(data.get("pixel_count", 0)) for _, data in graph.nodes(data=True)],
            dtype=np.int64,
        )
        if has_pixel_counts
        else None
    )
    metadata_raw = graph.graph.get("metadata", "{}")
    try:
        metadata = json.loads(metadata_raw)
    except TypeError:
        metadata = {}
    neighbors, contact_map, centroid_map, pixel_count_map = _graph_from_arrays(
        nodes,
        edges,
        contacts=contacts,
        centroids=centroids,
        pixel_counts=pixel_counts,
    )
    return LabelGraphData(neighbors, contact_map, centroid_map, pixel_count_map, metadata)

