# UI/UX review — 2026-10-27

Reviewed the running application with Playwright on desktop, at 900px, and at 390px. Also exercised the actual plain-HTTP LAN URL. Applied [Vercel's Web Interface Guidelines](https://raw.githubusercontent.com/vercel-labs/web-interface-guidelines/main/command.md), using the installed `web-design-guidelines` skill.

## Resolved findings

- `frontend/src/ui.tsx:4` — LAN HTTP lacked `crypto.randomUUID`; adding nodes and saving crashed. IDs now use `crypto.getRandomValues`.
- `frontend/src/ui.tsx:10` — Copy code threw when Clipboard API was unavailable on LAN HTTP. Added a working copy fallback and error feedback.
- `frontend/src/main.tsx:859` — Invalid property edits silently failed; numeric overrides could not return to symbolic bindings. Added inline accessible errors and reversible `$width`-style bindings.
- `frontend/src/main.tsx:1186` — Undo discarded the current nested navigation. Preserve the deepest valid location in the restored graph.
- `frontend/src/ui.tsx:43` — Dialogs lacked Escape dismissal, focus containment, and focus restoration. Added all three and made background panels inert while a dialog is open.
- `frontend/src/main.tsx:1347` — Switching architectures could discard unsaved edits. Offer save/open, discard/open, or continued editing.
- `frontend/src/main.tsx:1389` — Saved-project deletion lacked recovery. Add an Undo action accessible inside the project dialog and from the notification.
- `frontend/src/main.tsx:2644` — Phone layouts hid both the block library and inspector. Add usable Add blocks, Models, and Properties drawers.
- `frontend/src/main.tsx:1292` — Automatic layout used fixed dimensions despite variable node sizes. Use measured node dimensions and correct center offsets.
- `frontend/src/style.css` — Small text and low-contrast semantic labels made controls and properties hard to read. Increase text sizes, strengthen contrast, enlarge ports, respect reduced-motion preferences, and provide visible focus states.
- `frontend/src/main.tsx` — Block search carried over to the model library and could hide all architectures. Reset search when switching libraries; explain empty block-search results.
- `backend/templates.py` — The recurrent transformer card overlapped its following normalization card. Increase spacing in the predictor example.

## Verification

- Production TypeScript/Vite build passed.
- The 18-test Playwright regression suite passed after the UI fixes, including eight new UX tests.
- The additional browser regression for overlap-free predictor cards before and after automatic layout also passed: 19 passing browser checks in total.
- Axe checks reported no WCAG A/AA rule violations in the main workbench and tokenizer dialog.
- Actual LAN HTTP walkthrough: add/delete node, save project, copy code, and nested navigation completed without browser JavaScript errors.
- Desktop and phone screenshots are under `data/ux-audit/`.

## Hosting

The running server and `scripts/start.sh` default to `0.0.0.0:8173`. The current LAN URL is **http://192.168.50.172:8173**. `WEAVE_HOST=127.0.0.1` remains available as an override.

No additional Arch system packages were needed. Installed the UI review skill and added `@axe-core/playwright` as a development dependency.

## Composable computation update

- Distinct motifs now communicate recurrent transformer applications, masked-token refinement, independently learned adapters, n-gram memory, and attention patterns. Transformer stage summaries derive from their actual body.
- Explicit applications show `θ` groups and dashed parameter-sharing links. Editing a shared body updates its other applications; making a copy independent separates its parameters.
- Tensor connections open an insertion inspector. Loops can expand into editable applications, preserving shared/independent parameters and sampling schedules.
- Sparse patterns are ordinary graph nodes with union/intersection. Inspector fields follow the selected pattern, custom matrices have clickable cells, and optional mask inputs are marked.
- TypeScript strict mode is enabled. Removed the broad `Any` alias, added unknown-boundary parsing, and extracted typed registries and graph rewrites. Prettier scripts keep these modules consistently formatted.
- Python verification: 31 passing checks, including output/parameter parity, shared module identity, independent adapter gradients, diffusion schedule parity, composed masks, and masked attention expansion. GPU FlexAttention remains unverified on this CPU host.
- Browser regressions include interleaved learned layers, propagation/independence of shared definitions, loop expansion, sparse settings, shared application creation, and clickable/custom-invalid matrix editing.

## Arrow routing and phone layout review

Found by clicking through the examples at 1600 × 1000 and 390 × 844:

- `frontend/src/main.tsx` — Built-in endpoint-only smooth-step paths ran residual skips through attention/FFN cards, and Needle bypasses crossed multiple operations. `edges.tsx` and `routing.ts` now route around measured rectangles, with rounded orthogonal corners and distinct fan-out corridors. Weight-sharing links use the same obstacle geometry. Dragging any card recalculates its surrounding routes.
- `frontend/src/graph.ts` — Legacy narrow gaps could leave almost no room beyond a connection handle. Fresh examples now use measured widths, consistent stage heights, aligned ports, and wider rank/branch spacing. Shared definitions keep their computation while their instances remain freely positioned. Restored/imported coordinates are preserved until Auto layout is requested.
- `frontend/src/main.tsx`, `GraphNavigator.tsx` — Fitting an eight-stage horizontal graph into a phone viewport reduced cards to about 13% scale. Phone views now start at a readable node scale, with Previous/Next controls in computation order. Fit graph remains an explicit overview action. Focus restores the readable view, and following nodes does not open the properties drawer.
- `frontend/src/style.css` — Arrow strokes and pointer hit areas previously shrank with zoom. Routes now keep readable stroke widths and usable hit areas, with direction markers and shape labels on hover/selection. Selecting a card emphasizes its connected wires. The phone navigator and existing zoom/minimap controls occupy separate areas.

Automated routing coverage checks node overlap and actual rendered SVG paths against unrelated card interiors across all **73 example scopes**, including all Needle layers, on both desktop and phone. Additional checks cover dragging, backward links, weight sharing, tight legacy gaps, expanded attention with output gates/masks, independent-layer insertion, phone control overlap, and preservation of restored coordinates. Screenshots live in `data/routing-audit/`.

Final verification: **31 Playwright tests passed** (including all 73 scopes at each viewport), **6 routing unit tests passed**, strict TypeScript/Vite production build and formatting checks passed, and the service remains healthy on `0.0.0.0:8173`. The main workbench/tokenizer accessibility regressions also passed.


## Label placement and ELK arrangement review — 27 October 2026

The previous path-only audit did not cover the full rendered text rectangles. This pass enables all labels while opening each model scope, and checks label/card and label/label intersections alongside node overlap and sampled SVG paths.

Fixed findings:

- `frontend/src/edges.tsx:172` — Centre-only label placement could cover nearby cards. Labels now register their measured dimensions and use shared, deterministic placement against full node and label rectangles.
- `frontend/src/layout-core.ts:137` — The previous layout integration did not reserve edge-label dimensions or preserve engine-produced routes. ELK now receives measured labels and fixed ports, and its node/edge geometry is retained together.
- `frontend/src/main.tsx:2232` — Automatic layout was difficult to discover. Auto arrange now has a visible text label and pending state; Show edge labels is an accessible toggle. Phone toolbar actions wrap without horizontal overflow.
- `frontend/src/main.tsx:1064` — Delayed initial arrangement could move newly edited graphs. Recorded edits cancel pending initial arrangement; asynchronous results and delayed fitting check the edit revision.

Research and implementation tradeoffs are documented in [LAYOUT_RESEARCH.md](LAYOUT_RESEARCH.md). The graph canvas is React Flow; Weave is the custom application name. ELK performs automatic arrangement in a worker. Existing manual positions remain editable and are preserved on restore/import.

Verification for this pass: **33 Playwright tests passed**, including every one of the 73 scopes on desktop and phone with all available labels visible, long labels at multiple zoom levels, drag rerouting, undo, startup edit cancellation, and accessibility checks. **13 geometry/layout unit tests passed**. Strict TypeScript, production build, formatting and LAN health checks passed. Service: `0.0.0.0:8173`, reachable at `http://192.168.50.172:8173`.

## Split references and shareable workspace URLs

- `frontend/src/ReferencePane.tsx` — A reference uses a cloned document and its own React Flow provider. Nested blocks, shapes, parameter groups, node navigation, zoom, label visibility, and temporary arrangement use the same presentation/routing as the editable graph. Read-only properties expose settings without editing them. Reference keyboard events cannot delete or undo the working graph.
- `frontend/src/SplitDivider.tsx` — Pointer resizing has a keyboard alternative, bounded proportions, visible focus, and an equal-width reset. Phones use stacked, scrollable panes. Explicit pixel padding keeps focused cards clear of navigator controls; numeric padding had initially reduced them to unreadable scale.
- `frontend/src/workspaceLocation.ts` — Typed URL parsing/serialization restores the model, nested blocks, selection, tab, library/drawer, split state, reference location, and pane ratio. Repeated block parameters preserve arbitrary IDs. View updates replace the current history entry. Model documents remain in server saves or local recovery.
- `frontend/src/main.tsx`, `graphPresentation.ts` — Both views share node/edge decoration while retaining independent selections and viewport stores. An explicit main-flow ID keeps accessibility descriptions distinct across canvases. Missing saved references have a retry/replacement path; a failed primary-model link can return to the default workspace.

Browser coverage includes editing while browsing, read-only dragging/shortcuts, independent zoom, copied URLs in fresh contexts, saved-reference snapshots, pane resizing, phone stacking, missing references, library-drawer restoration, and automated split-view accessibility checks. Responsive screenshots were reviewed at 390 and 1600 pixels; horizontal overflow was also checked at 320, 900, and 1200 pixels.

The existing walkthroughs now scope model selection to the example library and give deletion/undo fixtures unique names, so tests do not confuse saved user models with examples or delete another project's data.

Final validation for the split/URL update: **40 browser checks passed** across the full regression run and focused reruns, **42 Python tests passed**, and **17 layout/URL unit tests passed**. Strict TypeScript, production build, formatting, split-view accessibility, and LAN health checks passed. The original GitHub MIT license/history is preserved, and local runtime data, dependencies, compiled assets and browser artifacts are excluded from version control.
