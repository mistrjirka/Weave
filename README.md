# Weave · Neural Graph Studio

A self-hosted browser workbench for composing neural networks and exporting readable, runnable PyTorch. Attention, embeddings, n-grams, memory, loops, and sampling are compatible graph building blocks. Split view lets you edit a network alongside a reference architecture.

## Install and run

Python 3.11+ and Node 20.19+ are required.

```bash
git clone https://github.com/mistrjirka/Weave.git
cd Weave
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
npm ci
npm run build
bash scripts/start.sh
```

For an existing installation, run `bash scripts/start.sh` from the repository root. The server listens on **0.0.0.0:8173** by default. Open **http://127.0.0.1:8173** on the server, or `http://YOUR_SERVER_IP:8173` from another device on the LAN.

SSH forwarding is also supported:

```bash
ssh -L 8173:127.0.0.1:8173 USER@SERVER
```

Then open the localhost URL in that computer's browser. `WEAVE_HOST`, `WEAVE_PORT`, and `WEAVE_DATA` configure listening and storage. Set `WEAVE_HOST=127.0.0.1` for localhost-only access. This is a single-user service without built-in authentication.

PyTorch is required to run exported models and the computational tests, but is not needed for graph editing or code generation:

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For frontend development, run the API on port 8173 and `npm run dev` in another terminal. Vite serves port 5173 and proxies `/api` to the backend.

## Design a model

- The default composition appends mask tokens, then repeats a token predictor and confidence selection. Open its loop body, then the predictor, to see embeddings and n-gram memory feeding a shared-weight transformer loop.
- Click a library entry to add a node. Connect its labeled input ports by dragging from another node's output. Select edges or nodes and press Delete to remove them.
- Double-click a composite node or choose **Open block / Open loop body** to edit its internals. Breadcrumbs return to enclosing graphs.
- Select compact attention and choose **Edit attention internals** to expose editable Q/K/V projections, head transforms, RoPE, normalization, attention, and output gating. This preserves its computation and parameter count.
- Change model configuration in the right inspector. Properties such as `$width`, `$heads`, `$ffn_width`, and `$vocab` bind to that configuration. Numeric overrides replace bindings. Containers can expose attention and FFN settings as per-instance overrides.
- **Loop container** repeats an arbitrary subgraph. Input `a` is loop state; additional inputs are loop-invariant values. The output must preserve state shape and dtype. Shared weights reuse one body; independent layers allocate one body per iteration.
- Shift-click computation nodes and press **Ctrl+G** to package them as a reusable block. Blocks currently expose multiple inputs and one output. Save a selected block to your local block library.
- A custom Python block defines `CustomBlock(nn.Module)` and declared input/output shapes. Optional `input_shapes` exposes multiple ports; `input_dtypes` and `parameter_count` describe additional contracts. The server validates syntax and contracts without executing user source. User code runs only when you run its exported package.
- Save persists projects on the server. Browser-local recovery preserves the current project. Project settings provide JSON download; Import restores a complete project.
- On narrow screens, **Add blocks**, **Projects**, and **Properties** open usable side panels. The zoom percentage button returns the graph to 100% for reading node details.
- Invalid property edits show an inline error and leave the last applied value intact. Numeric overrides can be changed back to `$width`-style bindings. Undo preserves your current position inside nested graphs.
- Switching models with unsaved changes offers Save & open, Discard & open, or Keep editing. Dialogs support Escape and keyboard focus navigation.

## Tokenizers and dimensions

Choose a Hugging Face repository (with optional revision), server-local path, or uploaded tokenizer bundle. Supported inputs include `tokenizer.json`, SentencePiece `.model`, and standard Transformers bundles with vocabulary/merges/configuration files. Remote custom Python is disabled.

The inspector reports vocabulary cardinality, base vocabulary, special tokens, and required embedding capacity (`max(token IDs) + 1`). Added tokens are included. An actual tokenization preview shows IDs and `[batch, sequence]` dimensions. A bare tokenizer model may not contain semantic special-token roles; upload its tokenizer configuration as well when available.

Embeddings and vocabulary heads bound to `$vocab` update when a tokenizer is selected. Mask IDs update when the tokenizer defines a mask token. Otherwise the model's mask ID remains editable and must be a valid ID for the intended sampling scheme. Tokenizer length limits are metadata, not inferred network context limits.

Shape validation propagates symbolic batch/sequence dimensions through nested blocks, n-gram paths, and loops. It checks residuals, concatenation, attention heads, dtype, vocabulary capacity, lookup buckets, tied heads, missing connections, and direct cycles. Errors identify the failing node and port. Unsupported or unresolved computation blocks runnable export.

## Reference projects and fidelity

| Project | Included computation | Limits |
| --- | --- | --- |
| Decoder transformer | RoPE, MHA/GQA/MQA, causal/windowed attention, RMSNorm, SwiGLU, residual paths | Random weights; no training/inference service |
| Looped transformer | Explicit shared-weight recurrent depth | Fixed iteration count |
| Text diffusion | Bidirectional predictor, append-mask operation, confidence/random selection, linear unmasking loop | LLaDA-style composition, not the original 8B architecture/checkpoint; no CFG |
| Needle 1 | 12-layer attention-only encoder, 8-layer self/cross-attention decoder, gated residuals, shared embeddings | Documented topology port; no original QAT, initialization, cache or checkpoint conversion |
| Needle 2 | Published 27-layer settings; Engram sites, gated attention, learned Monarch mixer, four-lane mHC routing | Uses current SAN computation with Needle 2 config; historical checkpoint parity is not asserted |
| Needle 3 | Published 20-layer settings; asymmetric QK/value heads, causal convolution, global/local attention, Engram, Monarch, mHC | Full-depth backbone; no trained ladder slicing, probe heads, padding-aware packed masks or quantization |
| Thimble | 20-layer 448-wide trunk, gated GQA 8/4, QK norm, sandwich norms, SwiGLU, tied embeddings | Grammar dispatcher, retrieval, auxiliary name/pointer heads and pretrained weights are outside the trunk graph |

Upstream settings and pinned revisions are available in **References** and retained in project JSON. Float32 PyTorch ports do not claim bitwise parity with quantized deployment engines. The Engram hash and Monarch equations are checked against independent NumPy translations; attention is checked against the saved upstream Thimble implementation.

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

Choose **New project → Start from → Looped transformer** to create a project with three applications of one transformer module, with independently learned linear layers between them. Purple `θ` links indicate parameter identity, rather than a tensor connection. Open any application to edit its graph: the other members of its weight group receive the same definition. **Make this application independent** copies its definition into a separate parameter set. Ordinary Duplicate also creates independent parameters; **Add application with shared weights** creates another application of the same module.

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

The address bar tracks the current model, open blocks, selected node, Architecture/PyTorch code/References tab, library/drawer, split state, reference model and block, and pane proportions. Copy the address to reopen that view. For example:

```text
/?model=transformer&block=stack&split=1&reference=example%3Aneedle1&refBlock=decoder
```

Saving a network gives it a server project ID that links can reopen on the same server. Local recovery preserves unsaved edits in the current browser; a URL contains view state rather than a copy of those edits. Save the model before sharing edits with another browser. Reference views use snapshots: Reload reference fetches the latest saved version. No project data or tokenizer contents are embedded in the URL.

The application is MIT licensed; see [LICENSE](LICENSE). Third-party architecture sources, libraries and fonts retain the licenses documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Design assistance

The validation bar reports tensor errors separately from **Design notes**. Notes also appear in the selected block's properties, with explanations, suggested checks, and source links. They cover extreme hidden bottlenecks, consecutive linear projections, output softmax and cross-entropy, unused learned branches, discrete sampling gradients, fixed zero scales, and empty custom attention rows. They never block export. See [the research, exact heuristics, exceptions, and correctness tests](DESIGN_ADVICE.md).


## Creating, opening and studying networks

Use **New project** in the top bar to name a network and start from a blank canvas or an example. Creation makes an editable local draft with its own ID; **Save** keeps it on this server. The current draft also appears in the project browser so you can continue editing it. Unsaved status survives local recovery after reload.

**Your workspace** opens a browser with **Your projects** and **Reference architectures**. Saved projects appear first in the sidebar's **Projects** collection. **View reference** opens a read-only example beside your work; **Use as starting point** creates an independent editable copy. The reference pane also has **Use this example**. Viewing references does not replace your editing document.

Browser tabs and creation starters survive reload and copying the URL: `?workspace=projects`, `?workspace=examples`, or `?workspace=new&starter=looped`. See the [workflow assessment and tested user journeys](PROJECT_WORKFLOW_REVIEW.md).
