# Weave · Neural Graph Studio

A self-hosted browser workbench for composing neural networks and exporting readable, runnable PyTorch. Attention, embeddings, n-grams, memory, loops, and sampling are compatible graph building blocks. Split view lets you edit a network alongside a reference architecture.

## Install and run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), Node.js 20.19+ and npm. The installer uses uv to obtain Python 3.12, sync the committed lockfile, build the browser UI, and create a separate CPU training environment.

```bash
git clone https://github.com/mistrjirka/Weave.git
cd Weave
bash scripts/install.sh
bash scripts/start.sh
```

Open **http://127.0.0.1:8173**, or `http://YOUR_SERVER_IP:8173` on your LAN. The default address is **0.0.0.0:8173**. This is a trusted single-user workbench: browser-launched scripts run with the server user's permissions.

Training is enabled by the default installer. For a smaller editor/export-only installation, use `bash scripts/install.sh --training none`. To use an existing CUDA/ROCm environment, use `bash scripts/install.sh --training-python /absolute/path/to/venv/bin/python`; the installer keeps that environment's PyTorch build. See [installation, GPU setup, updates and reboot startup](INSTALL.md).

Settings persist in `.weave-config.json`, so restarting requires **no exported environment variables**:

```bash
bash scripts/configure.sh --projects-dir ~/shared/weave-projects --port 8173
bash scripts/configure.sh --enable-training --training-python /path/to/training/bin/python
bash scripts/start.sh
```

Project folders have readable names such as `hello-there`; renaming a project renames its folder. Internal databases/run history default to `data/`, independently of editable projects. Environment variables remain optional deployment overrides; `.env.example` lists them, but `.env` files are not automatically loaded. Restart after configuration changes.

For frontend development, run the API on port 8173 and `npm run dev` in another terminal. Vite serves port 5173 and proxies `/api` to the backend.

## Design a model

- The default composition appends mask tokens, then repeats a token predictor and confidence selection. Open its loop body, then the predictor, to see embeddings and n-gram memory feeding a shared-weight transformer loop.
- Click a library entry to add a node. Connect its labeled input ports by dragging from another node's output. Select edges or nodes and press Delete to remove them.
- Double-click a composite node or choose **Open block / Open loop body** to edit its internals. The **Up one level** button and breadcrumbs return to enclosing graphs, independently in each split pane.
- Select compact attention and choose **Edit attention internals** to expose editable Q/K/V projections, head transforms, RoPE, normalization, attention, and output gating. This preserves its computation and parameter count.
- The movable **Variables** card is visible on the canvas by default. Edit values there, add variables, or open **Graph variables** for the full table. Enter `$name` in a block field to bind it, or a number for an independent value; bound fields show their resolved value. Variables inside a nested graph override inherited variables and enclosing block parameters. Attention and FFN settings remain configurable on each block.
- **Looped transformer** in the block library creates a premade recurrent transformer. Its outer inspector exposes heads, KV heads, FFN width, causal settings, and an **Attention type** dropdown for MHA, GQA, MQA, sliding-window, or local/global sparse attention. The dropdown updates the inner editable graph while preserving other layers.
- **Loop container** repeats an arbitrary subgraph. Input `a` is loop state; additional inputs are loop-invariant values. The output must preserve state shape and dtype. Shared weights reuse one body; independent layers allocate one body per iteration.
- Drag a rectangle on empty canvas to select nodes, or **Ctrl/Cmd-click** to add or remove individual nodes (Shift-click also works). **Ctrl/Cmd+C** copies selected nodes, their internal connections, and nested blocks; **Ctrl/Cmd+V** pastes an independent editable copy into the current graph. Repeated pastes are offset. Normal text copy/paste remains available in property fields. Hold **Space** while dragging to pan, or use the middle/right mouse button; touch gestures still pan.
- Select computation nodes and press **Ctrl/Cmd+G** to package them as a reusable block. Blocks currently expose multiple inputs and one output. Save a selected block to your local block library.
- A custom Python block defines `CustomBlock(nn.Module)` and declared input/output shapes. Optional `input_shapes` exposes multiple ports; `input_dtypes` and `parameter_count` describe additional contracts. The server validates syntax and contracts without executing user source. User code runs only when you run its exported package.
- Save persists projects on the server. Browser-local recovery preserves the current project. Project settings provide JSON download; Import restores a complete project.
- On narrow screens, **Add blocks**, **Projects**, and **Properties** open usable side panels. The zoom percentage button returns the graph to 100% for reading node details.
- Invalid property edits show an inline error and leave the last applied value intact. Numeric overrides can be changed back to `$width`-style bindings. Undo preserves your current position inside nested graphs.
- Switching models with unsaved changes offers Save & open, Discard & open, or Keep editing. Dialogs support Escape and keyboard focus navigation.

## Tokenizers and dimensions

Select a token Input node, then choose **Select tokenizer** in its properties. Each input can use its own Hugging Face repository (with optional revision), server-local path, or uploaded tokenizer bundle. Token inputs inside connected blocks inherit their tokenizer from the originating input. Supported inputs include `tokenizer.json`, SentencePiece `.model`, and standard Transformers bundles with vocabulary/merges/configuration files. Remote custom Python is disabled.

The inspector reports vocabulary cardinality, base vocabulary, special tokens, and required embedding capacity (`max(token IDs) + 1`). Added tokens are included. An actual tokenization preview shows IDs and `[batch, sequence]` dimensions. A bare tokenizer model may not contain semantic special-token roles; upload its tokenizer configuration as well when available.

Connected embeddings and vocabulary heads bound to `$vocab` automatically use the originating input tokenizer’s capacity, including added token IDs. This binding follows nested blocks and loops; cross-attention preserves the decoder input’s vocabulary. Embedding feature width is a separate per-block property. Numeric vocabulary overrides are validated against that input’s token IDs. Mask IDs follow the tokenizer when it defines a mask token. Otherwise the model's mask ID remains editable and must be a valid ID for the intended sampling scheme. Tokenizer length limits are metadata, not inferred network context limits.

Exports include each configured input’s bundle under `tokenizers/<id>/`, plus `tokenizers.json` and `load_tokenizer(input_node="source")` for selecting between inputs. Existing project-level tokenizer settings are migrated onto their token inputs.

Shape validation propagates symbolic batch/sequence dimensions through nested blocks, n-gram paths, and loops. It checks residuals, concatenation, attention heads, dtype, vocabulary capacity, lookup buckets, tied heads, missing connections, and direct cycles. Errors identify the failing node and port. Unsupported or unresolved computation blocks runnable export.

## Reference projects and fidelity

| Project | Included computation | Limits |
| --- | --- | --- |
| Decoder transformer | RoPE, MHA/GQA/MQA, causal/windowed attention, RMSNorm, SwiGLU, residual paths | Random weights; no training/inference service |
| Looped transformer | Explicit shared-weight recurrent depth | Fixed iteration count |
| Text diffusion | Bidirectional predictor, append-mask operation, confidence/random selection, linear unmasking loop | LLaDA-style composition, not the original 8B architecture/checkpoint; no CFG |
| Needle 1 | 12-layer attention-only encoder, 8-layer self/cross-attention decoder, gated residuals, shared embeddings | Documented topology port; no original QAT, initialization, cache or checkpoint conversion |
| Needle 2 | Published 27-layer settings; Engram, gated attention, two-stage Hadamard mixer, four-lane mHC routing | Historical float32 backbone; no checkpoint conversion, packed masks or quantization |
| Needle 3 | Published 20-layer settings; asymmetric QK/value heads, causal convolution, global/local attention, Engram, Monarch, mHC | Full-depth backbone; no trained ladder slicing, probe heads, padding-aware packed masks or quantization |
| Thimble | 20-layer 448-wide trunk, gated GQA 8/4, QK norm, sandwich norms, SwiGLU, tied embeddings | Grammar dispatcher, retrieval, auxiliary name/pointer heads and pretrained weights are outside the trunk graph |

Upstream settings and pinned revisions are available in **References** and retained in project JSON. Float32 PyTorch ports do not claim bitwise parity with quantized deployment engines. Weight-matched small-model logits are compared with pinned upstream JAX for Needle 1/2/3, upstream PyTorch for Thimble, and Transformers for GPT-2, Hammer and SHELLper. These checks cover float32 full-sequence inference, not trained checkpoints or deployment kernels.

The generic n-gram lookup and fixed Hadamard mixer are separate from architecture-specific Engram and learned Monarch operations. They can all be combined freely in custom graphs.

## Export

**Export PyTorch** downloads a ZIP containing `model.py` with descriptive class/layer/input names, explicit PyTorch constructors, direct tensor expressions, and ordinary `for` loops. Independent repeated layers use `nn.ModuleList`; shared repetitions reuse a named module. Weight tying and explicit parameter groups preserve module identity. Equivalent nested definitions reuse generated classes. Specialized layers remain in `runtime.py`, tensor helpers in `export_helpers.py`, and attention-mask support in `sparse_runtime.py`. These are ordinary Python/PyTorch files shipped together; generated models use no graph interpreter, `make_module`, runtime `Loop`, or numbered module dictionary. The ZIP also includes project JSON, dependencies, `example.py`, tokenizer loading, selected tokenizer assets, and source attribution. Weights start randomly initialized. Run:

```bash
python -m pip install -r requirements.txt
python example.py
```

Generated code is formatted with Black on the server; exported models do not depend on Black. Input and layer names come from graph labels, with safe identifiers and automatic handling of duplicate/reserved names. Differentiable prediction graphs support standard PyTorch autograd; training requires data, a loss and an optimizer. Discrete sampling nodes are inference operations. New exports use descriptive `state_dict` keys; checkpoints from older numbered exports require key mapping. Export regression checks cover forward execution, parameter counts and sharing, compact/unrolled sampling parity, safe Python identifiers, optional input positions, and loss/backward/optimizer/checkpoint reloads for the seven prediction-model examples.

The example uses small batch/sequence dimensions but retains the graph's feature widths and layer counts. Reference architectures can therefore allocate their full parameter count. The automated tests use reduced configurations.

## Verify

```bash
.venv/bin/python -m pytest tests/test_graph.py -q
npx playwright install chromium
npm run build
npx playwright test
```

Browser tests cover hierarchical editing, visible recurrence, nested shape errors, tokenizer upload/preview, source examples, code export, save/reload, custom blocks, and narrow layouts. Screenshots are written under `data/`.

## Files and API

`backend/core.py` owns contracts and shape inference; `backend/compiler.py` generates classes; `backend/runtime.py` supplies PyTorch operations. React Flow renders the same graph document used by the backend. Projects use schema version 1.

- `GET /api/templates`, `GET /api/projects`
- `GET`, `PUT`, `DELETE /api/projects/{id}`
- `POST /api/validate`, `/api/code`, `/api/export` accept a project document
- `POST /api/tokenizers/inspect` accepts `{source, revision}`
- `POST /api/tokenizers/upload` accepts multipart tokenizer files
- `POST /api/tokenizers/preview` accepts `{id, text}`

SQLite projects and tokenizer assets live under `data/`. Generated build output lives under `dist/`.

## Attribution

Architecture references: Cactus Compute Needle (`b4fc11b04f1852888d375c8c6dfbaca44572d014`), nikshepsvn Thimble (`bb7d5768651d525516f3647d551005a4411845af`), and ML-GSAI LLaDA (`9182493720ed723ef8031210d85959364e51cbe0`). Saved source excerpts and their upstream licenses are in `references/`. See `THIRD_PARTY_NOTICES.md`.

## Editing shared computation and sparse attention

Choose **New project → Start from → Looped transformer** to create a project with three applications of one transformer module, with independently learned linear layers between them. Purple `θ` links indicate parameter identity, rather than a tensor connection. Open any application to edit its graph: the other members of its weight group receive the same definition. Choose **Independent weights** in the **Weight group** selector to separate an application. **New weight group** creates a group; the selector offers only compatible node types. Joining a group adopts its definition and settings. Ordinary Duplicate also creates independent parameters; **Add application with shared weights** creates another application of the same module.

Drag an output onto empty canvas to open a searchable block picker; choosing a block creates and connects it in one undo step.

Select a tensor connection to **Insert independent layer** (linear, residual adapter, FFN, RMSNorm, or transformer). This splits the connection and preserves the existing weight groups. These are editable graph nodes, and can be deleted, rewired, nested, or saved in the block library. A compact Loop can be expanded into individual applications. Shared loops retain shared modules; independent loops retain separate modules. Expanded sampling steps retain their original step/total schedule, exposed in the inspector. Changing the number of sampling applications may require editing these schedules.

**Sparse attention composition** demonstrates `local ∪ global` feeding an attention mask port. The library also provides causal, full, block, deterministic random, content top-k, and custom boolean matrices, plus union/intersection. Attention's own causal/window restrictions intersect with the supplied pattern. Compact attention can be expanded to Q/K/V primitives while preserving its external pattern input. Custom matrices require explicit matching sequence lengths. Content top-k uses input features (or projected per-head Q/K when connected there), and computes dense routing scores; it is not a separate trained router.

The default **dense** backend uses PyTorch SDPA and materializes supplied patterns as boolean masks. This changes connectivity but does not guarantee sparse-kernel memory or speed benefits. **flex** uses compiled FlexAttention with block masks on a compatible CUDA GPU; **auto** selects it on CUDA and dense SDPA on CPU. GPU execution has not been verified on this CPU-only host. See [PyTorch FlexAttention documentation](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html).

The frontend uses TypeScript strict mode, explicit graph/parameter/API types, and Zod parsing at persisted/imported/API boundaries. `catalog.ts` defines library entries, `patterns.ts` defines pattern settings, `previews.tsx` registers computation motifs, and `graphOperations.ts` owns graph rewrites. Appearance never restricts graph connectivity. See the [TypeScript handbook on narrowing](https://www.typescriptlang.org/docs/handbook/2/narrowing.html).

## Following graph connections

Tensor arrows route through free space around the measured card boundaries, including residual skips, cross-attention inputs, backwards wires, and shared-weight links. Blue paths indicate routed bypasses; purple dashed paths indicate parameter sharing. Select a node to emphasize its incoming and outgoing wires, or select/hover a connection to see its tensor shape. Dragging a card recalculates the routes. Physically overlapping cards can leave no corridor; separate them or use **Auto arrange**.

Fresh examples use aligned stages and wider routing gaps. Restored/imported coordinates stay in place until you request **Auto arrange**. On phones, the canvas starts at a readable node scale; use **Previous node / Next node** to follow the computation without opening a properties drawer. **Fit graph** shows the entire overview, and **Focus current node** returns to the readable view. These are viewport controls over the same graph.

Run `npm run test:routing` for the routing geometry checks (Node 22.18+), or `npx playwright test tests/browser/routing.spec.ts` for desktop/phone walkthroughs of all example scopes, dragging, expanded attention, and coordinate preservation.


Auto arrange now runs ELK Layered in a browser worker using measured cards, fixed ports and tensor-label dimensions. It reduces crossings and reserves space for labels. **Show edge labels** displays tensor shapes together; manual dragging recomputes routes and places full label boxes clear of cards and other labels. When a crowded manual drawing has no free label location, select the connection for its details or use Auto arrange. Parameter-sharing links remain separate from computational ordering. See [layout research and tradeoffs](LAYOUT_RESEARCH.md).


## Split view and shareable locations

Use **Split view** above the canvas to edit your network on the left while browsing an example architecture or a saved network on the right. The reference has independent block navigation, zoom, label visibility, temporary arrangement and read-only property inspection. Reference interactions do not change your working document or undo history. Resize the desktop panes with the divider, or focus it and use Left/Right arrow keys; Enter restores equal widths. On phones the panes stack in a scrollable area.

The address bar tracks the current model, open blocks, selected node, Architecture/PyTorch code/Sources tab, library/drawer, split state, reference model and block, and pane proportions. Copy the address to reopen that view. For example:

```text
/?model=transformer&block=stack&split=1&reference=example%3Aneedle1&refBlock=decoder
```

Saving a network gives it a server project ID that links can reopen on the same server. Local recovery preserves unsaved edits in the current browser; a URL contains view state rather than a copy of those edits. Save the model before sharing edits with another browser. Reference views use snapshots: Reload reference fetches the latest saved version. No project data or tokenizer contents are embedded in the URL.

The application is MIT licensed; see [LICENSE](LICENSE). Third-party architecture sources, libraries and fonts retain the licenses documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Design assistance

The validation bar reports tensor errors separately from **Design notes**. Notes also appear in the selected block's properties, with explanations, suggested checks, and source links. They cover extreme hidden bottlenecks, consecutive linear projections, output softmax and cross-entropy, unused learned branches, discrete sampling gradients, fixed zero scales, and empty custom attention rows. They never block export. See [the research, exact heuristics, exceptions, and correctness tests](DESIGN_ADVICE.md).


## Creating, opening and studying networks

Use **New project** in the top bar to name a network and start from a blank canvas or an example. Creation makes an editable local draft with its own ID; **Save** keeps it on this server. The current draft also appears in the project browser so you can continue editing it. Unsaved status survives local recovery after reload.

**Your workspace** opens a browser with **Your projects** and **Reference architectures**. Saved projects appear first in the sidebar's **Projects** collection. Reference shortcuts are single rows; **Browse all examples** opens descriptions and copy actions. **View reference** opens a read-only example beside your work; **Create copy** creates an independent editable copy. The reference pane also has **Use this example**. Viewing references does not replace your editing document.

Browser tabs and creation starters survive reload and copying the URL: `?workspace=projects`, `?workspace=examples`, or `?workspace=new&starter=looped`. See the [workflow assessment and tested user journeys](PROJECT_WORKFLOW_REVIEW.md).

## Color themes

Choose **System**, **Light**, **Forest**, or **Black** with the theme selector beside Split view. On phones it appears as a monitor, sun, or moon icon. System follows your operating-system preference (Forest when dark); an explicit choice persists in this browser, survives reload, and synchronizes between open tabs. Theme selection does not change the graph or its shareable URL.

The dark palette covers node diagrams, loops, attention patterns, graph labels, properties, validation notes, generated code, project dialogs, and the reference pane. `frontend/src/theme.css` defines the shared palette; existing light colors remain the component fallbacks. Theme initialization runs before the app loads to avoid a light flash. Browser tests cover system changes, stored preferences, unavailable storage, desktop/phone contrast, and example-scope walkthroughs in each palette.

Code generation tracks the computational graph separately from its layout. Moving or measuring nodes, arranging nested graphs, and changing themes do not restart an in-flight compile; graph and parameter edits still regenerate code and shape diagnostics.

## Workspace clarity

The unselected inspector is a graph overview: trainable parameter count and shape-validation status. Select a block to edit that block, or a connection to insert a layer. Tokenizer selection has one entry point in the toolbar. **Sources** contains the current model's citations; **Reference architectures** is the example library.

**Project settings** (project title or sidebar footer) contains the name, description and project download. Architecture settings belong to blocks: transformer containers expose their own head count and FFN width; individual attention/FFN layers remain editable inside them. Existing files are loaded with legacy head/FFN defaults materialized onto the owning blocks, preserving their computation. Shared-weight applications still share their definition until made independent. New library layers start with concrete local defaults.

Dark mode separates the lighter forest-green library, raised panels, dark canvas and inset fields. The split divider has a full-height border and visible grip. Wide comparison panes open with the complete graph fitted; narrow panes retain node-by-node navigation, with **Fit graph** available for an overview. The top-level project name stays in the header rather than covering the canvas.

`tests/browser/workspaceReview.spec.ts` checks 3840×2160, 1920×1080 and 1280×1080 layouts, keyboard resizing, overview fitting, compact references, settings accessibility and independent transformer settings. Existing phone and all-example routing checks cover the smaller layouts and nested architectures. A migration regression verifies identical generated PyTorch source, inferred shapes, and parameter counts for every example before/after its defaults become block-local.

**Black** is the neutral dark option: black/charcoal workspace surfaces with colored computation diagrams. **Forest** retains the green dark palette. Both use the same remembered preference and pre-paint initialization.

Select reference blocks with Ctrl/Cmd-click or a drag rectangle, then copy with Ctrl/Cmd+C or **Copy reference selection** and paste into the editable pane. Copies preserve nested graphs and internal connections, use independent weight groups, and support undo. External dimensions are captured from the reference; local block variables and input-owned vocabulary bindings stay editable. Reference graphs remain read-only.

Split view has one active property selection. Selecting a reference block clears the editable highlight and displays its settings, output contract and parameter count in the right-hand inspector, labeled **Reference · read-only**. Selecting an editable block switches back to editing. Reference selection survives copied URLs and reloads; on phones a direct reference click opens the Properties drawer. The floating reference inspector has been removed.

Graph cards share a content gutter (`--node-inset` and `.node-body`) across their headers, previews, weight badges, and footers. Diagram components handle their internal layout without adding a second horizontal gutter. Loop and transformer stages wrap as connected groups; long labels remain inside the card. Browser walkthroughs check card bounds in every example scope alongside edge routing. `tests/browser/cards.spec.ts` also exercises every library entry, validation badges, and long denoising labels on phones; review galleries are written under `data/card-gallery-*.png`.

Run `npm run test:controls` to check typed weight groups and attention preset edits, including validation, exported PyTorch execution, gradients, and parameter counts for all five presets.

### Mixture of experts

The **MoE transformer** example shows an attention block and a separate MoE residual branch on the main canvas. Open the Mixture of experts block to see its router, expert blocks and weighted combine; open an expert to edit its graph. The outer block controls expert count, top-k, softmax or sigmoid router scoring, and the optional balancing-loss weight. New experts are independent copies. The fixed routing scaffold must stay connected as shown, while expert bodies can be edited.

For an always-on shared expert, add an **Always-on expert** block on a parallel branch from the same input and use an Add node to combine it with the MoE output. This is an ordinary editable graph branch. It is shared across tokens; use parameter groups separately if you want to share weights between layers.

For graphs with one floating-point output, the exported `train_example.py` demonstrates adding `moe_auxiliary_loss(model)` to a task loss. The placeholder loss and input must be replaced with training data and an appropriate objective. Selected tokens are packed into a single sequence for each expert, so attention or other token-mixing layers in an expert can mix unrelated sequences and lose original positions. The implementation does not impose expert capacity limits or provide distributed routing.

The LFM2.5 SHELLper reference includes editable GQA and double-gated causal-convolution layer graphs. Its exported backbone uses random weights and full-sequence inference; padding-aware attention, decoding caches, the tokenizer, and fine-tuned weights are not reproduced. Its 4,608-wide FFN reflects LFM2's automatic adjustment of the published 6,656 configuration value. Hammer 2.1 0.5B also has an editable random-weight Qwen2 backbone with QKV biases and YaRN; padding masks, KV cache, tokenizer and fine-tuned weights are not reproduced.

Sequence dimensions such as `T` are symbolic in the graph. They describe a variable input length, not a training context limit; an exported model's practical and trained context length depends on positional encoding, training data, and memory. Use a fixed input dimension when you specifically want the graph to require one length.

The block library also includes LayerNorm, GELU, ReLU, dropout, learned positional embeddings, and 1D/2D convolution and pooling. Sequence operators use `[B,T,C]`; 2D image operators use `[B,C,H,W]`. Fixed input dimensions are needed when a convolution or pool changes a symbolic spatial length. Dropout follows PyTorch train/eval behavior; learned position tables enforce their configured maximum length.

## Training workspace

The **Training** tab provides editable multi-stage PyTorch recipes, local launch or terminal monitoring, live metrics, generated examples, GPU monitoring, checkpoint lineage and verified snapshots. Single-device, DDP, supported split-model placement and separate evaluation-GPU strategies are available. Training execution is opt-in; access tokens are optional. Projects have on-disk folders under the configured projects directory (default `~/shared/weave-projects`), with saved graphs and editable training scripts. Generated network code, tokenizers and trainer support live separately under each stage’s `weave_support/` package. The default `train.py` contains typed Python configuration, a data-loader placeholder, a next-token loss example, plain AdamW and cosine decay with starting/minimum learning rates. Sequence lengths belong to your loader. Training can stop by epochs or optimizer updates, with separate work counters in the UI. Validation can run asynchronously on CPU or a selected GPU, using a bounded latest-checkpoint queue. HTTP reporting runs in a background thread. A checkpoint playground offers chat and bounded raw-text continuation on CPU or a selected GPU, recommending the saved checkpoint with the lowest matching validation loss. The optional bundled Trainer handles devices, accumulation, AMP, evaluation, reporting and checkpoints; new recipes need no user `config.json`. Snapshots retain source-file manifests and consumed-record traces without copying the corpus unless requested. Training displays SSH-friendly paths, the actual Python/PyTorch versions and a syntax-highlighted editor. Independent scripts can report through the HTTP API using the included connection file; localhost is the default. See [training setup, recipes, limitations, reporting API and backup/restore](TRAINING.md).

For Python recipe authors, see the [training authoring guide](TRAINING_GUIDE.md): custom loaders, token caching, augmentations, batching, progress, provenance and resume limitations.
