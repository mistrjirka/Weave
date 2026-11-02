# Tool catalogs and optional checkpoint-chat decoders

Checkpoint chat can pass a tool catalog to the saved tokenizer's chat template
and constrain token sampling with a trusted local Python module. This affects
inference only. It does not edit `train.py`, `weave_support/*`, the graph, model
weights, or training. Tool calls are displayed as text and never executed.

## Install and register a decoder

Install your decoder and its dependencies into the **checkpoint run's Python
environment** (shown in Training → Sources). A module can also live in a local
directory outside the project. For example:

```text
/home/you/serius-runtime/
    serius_runtime/
        __init__.py
        xml_grammar.py       # Your existing decoder, exposing build()
```

Add a `chat_decoders` entry to Weave's existing `.weave-config.json` at the
repository root, preserving its other settings:

```json
{
  "chat_decoders": {
    "xml-tool-grammar": {
      "label": "XML tool grammar",
      "target": "serius_runtime.xml_grammar:build",
      "python_path": "/home/you/serius-runtime",
      "schema_profile": "xml-flat-v1"
    }
  }
}
```

`python_path` is an absolute existing local directory containing the package;
omit it if the package is installed in the run's environment. `target` is a
Python module and factory name, not a URL or a source-code upload. Registration
is the explicit allowlist: the browser can select only an ID such as
`xml-tool-grammar`. Modules are imported only inside the isolated inference
worker, not the web server or a training process. These modules execute trusted
local Python; isolation is for inference lifecycle, not a security sandbox.

Choose **Refresh checkpoints** after editing the registry. Unload and reload an
already loaded model to make new registrations available to its worker.
An import or factory error is shown in the chat panel, with details in the
session's `chat/<session>/worker.log`. No packages or remote code are fetched.

## Use the browser controls

1. Open a run's **Chat & text completion** panel. Conversation and composer
   stay together; **Clear messages** and export sit beside the conversation.
   **Test settings** are beside chat on desktop and collapse on phone.
2. Choose **Mobile Actions (7 tools)**, **Generic demo**, upload a `.json` catalog,
   or paste a JSON list of function tools. The neutral empty editor is not a
   loaded example. The active badge and empty-catalog warning show the real state.
3. The Mobile Actions preset uses the supplied seven-tool catalog and its
   training system-prompt format with the browser's current local date/day.
   **Mobile Actions system prompt** can refresh that prompt independently.
   The date remains fixed for that conversation; clear messages to change it.
4. Choose **XML tool grammar** after registering your module. Choosing a decoder
   selects deterministic generation (temperature 0). You can intentionally change
   it afterward; each turn records temperature, seed and decoder.
5. **Load checkpoint** applies the draft settings. With an empty conversation,
   **Apply & send** saves changed settings before sending. Once a successful turn
   enters context, tools/system/decoder/template/format are pinned: choose
   **Clear messages & apply settings** to start a clean test with new settings.
6. Catalog, system, format and decoder defaults are saved in the project's
   `chat-settings.json`. Loading another checkpoint preserves those settings.
   **Clear messages** keeps the model/tools; **Unload model** releases weights.
7. Use **Conversation** to render tools through the saved template. **Text
   completion** continues the raw prompt without inserting history or tools;
   the decoder still receives the catalog. The context preview reports exact
   prompt tokens, the history contribution, reserved reply tokens and remaining
   space before generation. CPU inference is labeled as slow; an idle GPU is
   selected automatically only when the run's saved environment supports CUDA.

Example catalog (you can supply all seven tools in the same list):

```json
[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get weather for a city.",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": false
      }
    }
  }
]
```

Tool names must be unique, 1–64 characters, without control characters.
Dotted and Unicode names are accepted by the generic interface. Catalogs are
limited to 128 tools / 256 KiB; more than 10 tools gets a small-model/context
warning, not an arbitrary hard rejection. Consider retrieval/top-k selection.

Declare `schema_profile: "xml-flat-v1"` for this XML decoder. Weave then rejects
non-ASCII names/arguments and unsupported schemas **before loading/applying**,
instead of letting the decoder silently skip them. This profile supports flat
primitive string/integer/number/boolean arguments and scalar enums. Nullable/type
arrays, nested object/array constraints, anyOf/oneOf, $ref, patterns, formats,
numeric/string limits and other unenforced keywords are rejected explicitly.
Required names must be declared. The `generic` profile allows other decoders
with broader schema support. External schema references are never fetched.

The saved template must render `tools` to expose declarations to the model;
Weave uses the documented [Transformers tool-template interface](https://huggingface.co/docs/transformers/main/en/chat_extras).

## Decoder interface

```python
def build(tools, tokenizer, request):
    # Called once for every response, using that response's effective catalog.
    def mask(generated_ids, scores, request):
        # generated_ids: 1-D tensor, newly generated response tokens ONLY.
        # Initially empty; prompt/system/history IDs are not included.
        # scores: 1-D floating-point vocabulary logits, on the sampling device.
        # Return scores of the same shape/device; invalid tokens must be -inf.
        return scores
    return mask
```

Weave applies the mask **after repetition penalty and before greedy selection,
temperature and top-p sampling**. It rejects a wrong shape/device/dtype, NaNs,
positive infinity or a mask blocking every token. A new response gets fresh
processor state; the module factory is cached for the life of the worker.
XML spellings are preserved when a catalog or decoder is supplied. EOS and
recognized assistant turn-end markers stop generation and are omitted from the
normalized output. Historical assistant content strips trailing turn markers,
and parsed calls are passed as structured `tool_calls` to templates supporting
them. The template renders those calls once. Ordinary sampling without a
catalog/decoder retains its existing settings.

Your XML module must restrict tool names and argument keys to the supplied
catalog, escape values as needed, enforce balanced XML and allow `<no_call/>`.
It must permit tokenizer EOS when a response is complete. Weave does not
implement an XML grammar and cannot establish these guarantees for an arbitrary
plugin. The normal stop button and token/context limits remain in effect; a
response stopped mid-generation or cut off at its token limit may be an
incomplete valid prefix. Use a sufficient token budget and check finish reason.
Grammar constraints enforce syntax, not whether a tool or its arguments are
semantically appropriate.

## API

`GET /api/training/runs/{run}/chat` includes `decoders` (IDs, labels and
availability) and a session with `tools`, `decoder`, and `decoder_label`.
Module paths are not accepted in chat requests.

```http
POST /api/training/runs/{run}/chat/load
Content-Type: application/json

{"device":"cpu","decoder":"xml-tool-grammar","tools":[...]}
```

These become defaults for the loaded session. Apply different defaults without
reloading weights:

```http
PUT /api/training/runs/{run}/chat/settings
Content-Type: application/json

{"decoder":"xml-tool-grammar","tools":[...]}
```

Both endpoints require the existing local-execution/access rules. A generating
session must finish or be stopped before settings can be changed. Each response
uses a fixed catalog and decoder, recorded with its generation settings in
`chat/<session>/conversations.jsonl`.

```http
POST /api/training/runs/{run}/chat/messages
Content-Type: application/json

{"text":"What is the weather in Prague?","mode":"chat","max_new_tokens":256}
```

Omitted option fields inherit the loaded defaults. Explicit `null` clears
`tools`/`decoder`; `tools: []` supplies an explicitly empty catalog. A new session
with no saved project defaults starts with both None. Changes to pinned options
are rejected until messages are cleared. Conversation requests pass
`apply_chat_template(messages, tools=tools, ...)`.

## Parsed calls, context and simulated results

Each assistant turn shows raw XML, parsed tool/arguments, validation errors,
copy controls and the decoder/temperature/seed. Completed valid turns are marked
**In context**. Failed, empty, stopped or invalid turns remain visible but are
marked **Excluded from context**, including their paired user prompt. This
prevents displayed failures from silently becoming model history.

**Add mock result** opens a canned-result editor. Inserting it creates a
`role: "tool"` turn linked to the parsed call; the XML template renders it as a
`<result>` block. This is simulation only: no function or API is executed.
Tool/result continuation may be outside the checkpoint's training distribution.
The API endpoint is `POST /chat/tool-result` with `tool_call_id`, `content` and
`simulated: true`. Duplicate results for the same call are rejected.

**Export transcript** downloads JSONL containing messages, inclusion flags,
parsed calls, tools, decoder, checkpoint and generation settings. **Save as test
case** exports a user/assistant pair with the parsed expected call for review.
Server transcripts remain separately archived under the run's `chat/` folder.

## Regression tests

Expand **Regression tests**, import a `.jsonl` file or enter one object per line:

```jsonl
{"text":"Turn on the flashlight","expected":[{"name":"turn_on_flashlight","arguments":{}}]}
{"text":"Hello","expected":[]}
{"text":"Weather in Prague?","tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"expected":[{"name":"get_weather","arguments":{"city":"Prague"}}]}
```

Up to 50 independent cases run on the already loaded checkpoint. Each can supply
its own catalog, system, decoder and generation settings; otherwise it inherits
the session's settings. Tests default to temperature 0 and never execute tools.
`expected` can be a list of `{name, arguments}` calls, an XML string, or omitted
for an unscored exploratory case. Empty expected calls mean `<no_call/>`.

The table reports XML/schema parse validity, tool-name match and exact argument
match. Exact comparison uses parsed values, ignoring XML whitespace/attribute
formatting; it respects call order. Results export as JSONL, and durable JSON
records live in `chat/<session>/tests/`. The API is `POST /chat/tests` with
`{"cases":[...]}`. Starting tests does not replace the live conversation/context.

Checkpoint recommendations prefer same-step `eval/tool_exact_match`,
`eval/exact_match` or `eval/tool_accuracy` metrics when recorded. Otherwise the
fallback explicitly says that validation loss is not a tool-accuracy score.
Ad hoc suites with different cases are not silently used to rank checkpoints.

## Sessions and idle unload

Different runs have independent chat workers; loading one no longer unloads
another. Multiple tabs on the **same run** share that run's conversation.
Each loaded model uses additional CPU/GPU memory. The default idle timeout is
one hour; set `chat_idle_timeout_seconds` in `.weave-config.json` (60–86,400).
The UI displays the countdown. Idle unload preserves messages/settings; reloading
the same checkpoint can restore that conversation. A different checkpoint starts
new messages. A full web-server restart currently releases models/in-memory
sessions; project defaults and archived transcript files remain on disk.

Other API endpoints: `POST /chat/preview` computes tokens without a forward pass;
`POST /chat/validate-options` reports decoder schema errors/warnings;
`GET /chat/export` exports the current conversation.

## Verification

```bash
USE_TORCH=1 .venv/bin/python -m pytest tests/test_chat_workbench.py tests/test_chat_decoders.py tests/test_checkpoint_chat.py -q
npx playwright test tests/browser/checkpointChat.spec.ts
```

The integration tests use real frozen model exports/checkpoints and an
allowlisted dummy decoder forcing one token, covering greedy and sampled output,
fresh response state, tool-template forwarding, bad masks, catalog validation,
session defaults and unchanged source files. XML-specific guarantees must also
be tested against your installed grammar and tokenizer, including all declared
tools, forbidden tool/argument names, escaped values, `<no_call/>`, EOS and token
budget exhaustion. A forced-token dummy is not an XML-grammar correctness test.
