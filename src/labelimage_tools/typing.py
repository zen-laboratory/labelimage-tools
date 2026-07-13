"""Shared type aliases for label-image graph utilities.

The aliases here describe generic label-image adjacency/contact structures.
They intentionally avoid vertex-model-specific concepts such as edges, faces, or
``VertexModelGraph``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

Node = int | np.integer
Adj = Mapping[Node, Iterable[Node]]
Neig = dict[Node, np.ndarray]
Cont = dict[Node, np.ndarray]

LabelValue = int | np.integer | float | np.floating