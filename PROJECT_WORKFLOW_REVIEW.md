# Project workflow review

Applied the installed **web-design-guidelines** skill and the current
[Vercel Web Interface Guidelines](https://github.com/vercel-labs/web-interface-guidelines/blob/main/command.md).
This assessment uses code inspection and browser walkthroughs, not user interviews.

## Findings

Locations below refer to the previous interface at commit `175acfd`:

- `frontend/src/main.tsx:1998` — Blank project buried inside reference architectures; no explicit creation action.
- `frontend/src/main.tsx:2009` — Reference clicks replaced the editing document; preview and personal copy were not distinct actions.
- `frontend/src/main.tsx:2042` — Saved work followed the entire example collection.
- `frontend/src/main.tsx:2046` — Search did not filter saved work; timestamps were not localized.
- `frontend/src/main.tsx:1864` — “Neural workspace” hid the saved-project browser behind an unclear label.
- `frontend/src/main.tsx:2079` — Project title used a dropdown arrow but opened settings; a pencil now signals editing the name/configuration.
- `frontend/src/main.tsx:904` — Local recovery restored the graph without its dirty flag.
- `frontend/src/ui.tsx:62` — Mobile navigation was outside the modal's inert background.

## User journeys after the changes

| Intention | Action | Result |
| --- | --- | --- |
| Build a new network | **New project** → name → **Blank canvas** → **Create project** | A uniquely identified editable draft. The form explains how to start with input, blocks, and output. |
| Adapt an example | **Use as starting point** → name → **Create project** | An independent copy retaining the graph and source references. |
| Study an architecture | **View reference** | A read-only example beside the current editable graph, preserving unsaved work and history. |
| Adapt a viewed reference | **Use this example** | Creation form preselected to that example. |
| Resume saved work | **Your workspace** → **Your projects** → select a project | Opens the saved document, protecting current unsaved edits first. Selecting the current project returns to editing without reloading its saved snapshot. |
| Continue unfinished work | **Your projects** → **Current draft** → **Continue editing** | Returns to the current locally recovered document. |
| Persist a draft | **Save** | Stores it on the server using the same personal project ID. |
| Create while edits are unsaved | **Save & create**, **Discard & create**, or **Keep editing** | Explicit choice; a failed save leaves the old document and pending action intact. |

The sidebar puts personal work before reference cards. The full browser separates
**Your projects** and **Reference architectures**, with search, localized save
dates, useful empty states, and explicit viewing/copying actions.

Creating a project makes a local draft; **Save** persists it on the server. The
creation form, editor status, and draft card explain this distinction. Local
recovery preserves the graph and unsaved status. Share edits across browsers by
saving the project first.

## Accessibility and state

Labelled creation fields, inline errors with focus on the invalid name, collection
tabs with arrow/Home/End keys, focus management across dialog transitions, inert
mobile navigation behind modals, and accessible names containing visible button
text. Layouts adapt to 320px and 390px phones and desktop.

The URL retains the browser tab and creation starter:

```text
/?workspace=projects
/?workspace=examples
/?workspace=new&starter=looped
```

Existing graph/split-view URLs remain valid. Legacy `library=models` opens the
renamed Projects drawer. Draft names typed into the form are not in the URL.

## Verification

`tests/browser/projects.spec.ts` follows creation, local recovery, saving,
reopening, viewing references, independent copies, failed saves, keyboard control,
URL restoration, inline errors, empty collections and accessibility at 1600px,
390px and 320px. Existing editor tests use the same creation form to open editable
examples. Routing walkthroughs still cover every example scope on desktop and
phone. URL unit tests cover workspace tabs and starters.
