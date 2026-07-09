# labelimage-tools

`labelimage-tools` is a small utility package for 2-D labeled tissue
segmentation images. It provides loading, validation, preprocessing, adjacency
and contact extraction, junction detection, graph coloring, contours, and
matplotlib plotting helpers.

## Conventions

- Label images are 2-D NumPy arrays.
- Image coordinates are represented as `(y, x)`.
- The default background label is `0`.
- Labels do not need to be consecutive.
- Integer label values are preserved unless a function explicitly documents a
  relabeling operation.

## Installation

From this checkout:

```bash
python -m pip install -e .
```

For tests:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

## Load and preprocess labels

Use these helpers directly from scripts or notebooks. The intended starting
point is `load_image_pipeline(...)`: with its defaults, it crops foreground,
removes disconnected bits of repeated labels, and fills internal gaps. This
prepares the image so labels are clean, self-connected, unique regions that are
ready for adjacency, contour, and junction operations, with neighboring labels
touching across filled internal gaps rather than being separated by stray
background holes.

```python
import labelimage_tools as lit

labels = lit.load_image_pipeline("segmentation.tif")
```

## Adjacency and contact graph

Adjacency is computed by vectorized neighbor scanning. Contact values are
neighboring pixel-pair counts, useful as weights but not exact geometric
lengths. Original label IDs are preserved as graph node IDs.

```python
neighbors, contacts, centroids, pixel_counts = lit.graph_from_labels(labels)

lit.save_label_graph(
    "label_graph.npz",
    neighbors,
    contacts=contacts,
    centroids=centroids,
    pixel_counts=pixel_counts,
    source_image="segmentation.tif",
)

neighbors, contacts, centroids, pixel_counts, metadata = lit.load_label_graph(
    "label_graph.npz"
)

# JSON is a readable alternative for smaller graphs or inspection.
lit.save_label_graph("label_graph.json", neighbors, contacts=contacts)
```

## Junction detection

Junction pixels are pixels whose 3×3 neighborhood contains at least three
distinct labels. Connected junction pixels are clustered into `Junction`
objects with subpixel `(y, x)` centroids and the set of labels that meet there.

```python
junction_label_image, junctions = lit.junctions_from_labels(
    labels,
    background=None,
    min_labels=3,
    connectivity=2,
)

for junction in junctions:
    print(junction.id, junction.yx, sorted(junction.labels))
```

## Graph-colored plotting

Plotting helpers return matplotlib objects and never call `plt.show()`, so they
compose cleanly in notebooks.

```python
fig, ax = lit.plot_label_image(
    labels,
    use_graph_coloring=True,
    K=8,
    seed=1,
    title="Graph-colored labels",
)

fig, ax = lit.plot_junctions(labels, junctions=junctions, ax=ax)
```

You can also use the lower-level coloring helper:

```python
image, lut, ax = lit.show_map_with_colors(labels, K=8, seed=1)
```

## Examples and cookbook

The `examples/` directory contains script-style examples. Edit the constants at
the top of each script, run it, or copy sections into a notebook.

```bash
python examples/01_graph_coloring.py
python examples/02_preprocessing_gallery.py
python examples/03_junctions_and_contours.py
python examples/04_graph_io.py
```

The cookbook in [`docs/cookbook.md`](docs/cookbook.md) walks through the same
workflows and embeds the generated images.

### Essential processing outputs

Graph-colored label image using the cyclic `managua` colormap:

```python
labels = lit.load_image_pipeline("samples/test_cells2D.tif")

fig, ax = lit.plot_label_image(
    labels,
    use_graph_coloring=True,
    K=8,
    seed=4,
    cmap="managua",
    cyclic_cmap=True,
    title="Graph-colored label image",
)
```

![Graph-colored labels](examples/plots/graph_colored_managua.png)

Detected junctions:

```python
labels = lit.load_image_pipeline("samples/test_cells2D.tif")
junction_label_image, junctions = lit.junctions_from_labels(
    labels,
    background=0,
    min_labels=3,
    connectivity=2,
)
fig, ax = lit.plot_label_image(labels, cmap="managua", cyclic_cmap=True)
lit.plot_junctions(junctions=junctions, junction_mask=junction_label_image > 0, ax=ax)
```

![Detected junctions](examples/plots/junctions.png)

Ordered contours:

```python
labels = lit.load_image_pipeline("samples/test_cells2D.tif")
contours = lit.ordered_contours_from_labels(labels, background=0)
fig, ax = lit.plot_label_image(labels, cmap="managua", cyclic_cmap=True)
lit.plot_contours(labels, ax=ax, background=0, color="black", linewidth=0.6)
```

![Contours](examples/plots/contours.png)
