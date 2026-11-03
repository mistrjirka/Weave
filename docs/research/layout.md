# Graph layout and edge-label research

Reviewed 27 October 2026. Weave uses React Flow for interaction/rendering, with custom node cards, block diagrams and edge rendering. Layout is a separate subsystem.

## Why the label bug happened

The previous renderer checked whether a proposed label **centre** was inside a node. A clear centre could still leave half the text covering the card. It also placed each label independently, allowing labels to collide with one another. Routing a connector around a rectangle does not automatically place a readable text rectangle beside that connector.

React Flow's [edge-label tutorial](https://reactflow.dev/learn/customization/edge-labels) explains HTML label rendering and midpoint coordinates. Those coordinates are an anchor, not an obstacle-aware label-placement algorithm. Its [layout overview](https://reactflow.dev/learn/layouting/layouting) recommends separate layout engines.

## Established approaches

[yFiles documents two approaches](https://docs.yworks.com/yfiles-html/dguide/label_placement/):

- **Integrated labeling** gives label dimensions to the layout engine so it can create enough room while placing nodes and routes.
- **Generic labeling** places labels around an existing drawing without moving its nodes. This preserves manual work, but a crowded drawing may have no clear location for every label.

Weave uses both: integrated layout for Auto arrange, then measured candidate placement for visible labels, including after manual dragging. This also handles the purple parameter-sharing annotations without making them computational dependencies.

## Libraries considered

| Option | Relevant capability | Assessment for Weave |
| --- | --- | --- |
| React Flow | Interactive nodes, handles, edges and HTML label portal; integrates external engines | Keep as the browser canvas. It does not itself solve automatic placement. |
| Dagre | Fast directed layout, edge control points and label dimensions/positions | Kept for immediate draft positions. Our old integration supplied no label dimensions and used its own routes, leaving Dagre's label support unused. |
| ELK Layered / elkjs | Layer ordering, crossing reduction, orthogonal routing, constrained ports, edge labels and compound graphs | Selected for measured automatic arrangement. Browser worker support keeps the calculation off the UI thread. |
| libavoid | Interactive object-avoiding orthogonal/polyline routing | Useful for connector routing, but does not replace node layout and the full label-placement system. Browser integration would add a separate binding. |
| yFiles | Integrated and generic labeling, incremental layout and extensive diagram tooling | A capable commercial alternative. Switching the entire renderer/layout stack is unnecessary for these fixes. |
| Graphviz | Directed layout and post-layout external labels | Useful for generated/static diagrams; less natural for continuous React Flow editing. Its external-label controls illustrate the tradeoff between omitting a label and forcing an overlap. |

Primary documentation: [Dagre configuration](https://github.com/dagrejs/dagre/wiki#configuring-the-layout), [ELK Layered](https://eclipse.dev/elk/reference/algorithms/org-eclipse-elk-layered.html), [elkjs](https://github.com/kieler/elkjs), [libavoid](https://www.adaptagrams.org/documentation/libavoid.html), [yFiles licensing](https://docs.yfiles.com/yfiles-html/dguide/licensing/), [Graphviz external labels](https://graphviz.org/docs/attrs/xlabel/), [Graphviz forced labels](https://graphviz.org/docs/attrs/forcelabels/).

## Implementation

1. Measure actual node cards and edge text using the rendered CSS. DOM writes and reads are batched; ResizeObserver maintains rendered label sizes outside React render.
2. Give ELK each computation node, its fixed input/output ports, edges, and full label width/height. Use rightward layered orthogonal layout, layer-sweep crossing minimization and two-sided greedy switches. Return both node positions and edge bend points. See [ELK JSON](https://eclipse.dev/elk/documentation/tooldevelopers/graphdatastructure/jsonformat.html), [fixed ports](https://eclipse.dev/elk/reference/options/org-eclipse-elk-portConstraints.html), and [crossing strategy](https://eclipse.dev/elk/reference/options/org-eclipse-elk-layered-crossingMinimization-strategy.html).
3. Run elkjs in a separate browser worker. Preserve graph connectivity, node definitions, nested bodies and parameter sharing. A manual arrangement is undoable. Reject results if the graph's geometry or topology changed during the asynchronous calculation.
4. Cache returned routes against a signature of node positions/sizes and edge endpoints/ports. Selection and zoom do not invalidate geometry. Dragging or changing connectivity invalidates it and uses the existing obstacle-avoiding router.
5. Place each visible label as a full measured rectangle, testing nodes and previously accepted labels with a margin. Prefer ELK's reserved location, then candidates along and beside route segments. Selected labels have priority. If no candidate fits a fixed drawing, suppress that overlay; connection details remain available by selecting the edge and via its SVG title. Auto arrange creates additional room.
6. Keep parameter-sharing links separate from ELK's directed computation ordering. They are routed around cards and participate in label collision checks.

The **Auto arrange** button names its action explicitly. **Show edge labels** makes all available tensor annotations visible together. Fresh example scopes arrange after measurement; restored/imported coordinates remain unchanged until the user requests arrangement. Nested scopes are arranged independently as the user opens them; this is not a compound-graph rendering implementation.

## Limits and verification

Crossing reduction uses heuristics: arbitrary diagrams are not guaranteed to have zero crossings or a globally optimal layout. Purple sharing links are routed separately and are not included in ELK's crossing objective. The fixed-drawing label placer checks labels against cards and other labels, not every unrelated connector; dense wiring can still need manual inspection. Arbitrarily long labels can widen an arranged graph. Entire-model overviews may need zooming; phone navigation focuses readable individual blocks.

Automated geometry checks cover connector paths, full label rectangles against node rectangles, label-to-label overlap, and node overlap. The walkthrough opens every scope of every shipped model on desktop and phone with labels enabled. Separate unit checks cover long text, no-space suppression, priority, cache invalidation, and a deliberately crossed graph that ELK reorders. These checks establish regression coverage for the shipped examples, not a proof for every user-authored graph.


Example-scope coverage (each checked at 1600 × 1000 and 390 × 844):

| Model | Scopes |
| --- | ---: |
| Composed language model | 4 |
| Decoder transformer | 2 |
| Looped transformer | 4 |
| Masked text diffusion | 4 |
| Needle 1 | 3 |
| Needle 2 | 29 |
| Needle 3 | 22 |
| Thimble | 2 |
| Sparse attention composition | 2 |
| Blank project | 1 |
| Total | 73 |

The walkthrough also asserts that every available edge-label overlay is visible after arrangement, so passing collision checks cannot be achieved by hiding all labels. Root-model screenshots are saved as `data/routing-audit/elk-<model>-desktop.png` and `elk-<model>-phone.png`.

Final verification: 33 browser tests passed; 13 routing, label and ELK unit tests passed; strict TypeScript/Vite build and formatting checks passed. The browser run completed in 5.7 minutes on this host. No additional Arch system packages were needed.
