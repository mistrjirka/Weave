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
      "python_path": "/home/you/serius-runtime"
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

1. Open a run's **Chat & text completion** panel.
2. Expand **Tools & constrained decoding**. Upload a `.json` catalog or paste a
   JSON list of OpenAI-style function tools. Uploading fills the editor and does
   not activate it immediately. Leave the editor empty to remove tools.
3. Choose **None** for existing decoding or **XML tool grammar** after installing
   and registering your module. An unregistered XML decoder is marked
   **not installed**.
4. Load the checkpoint. If it is already loaded, choose **Apply tool settings**;
   this updates defaults without reloading model weights. The active badge shows
   the decoder and tool count. Sending is disabled while draft settings are
   invalid or unapplied.
5. Use **Conversation** to render tool declarations through the saved chat
   template. **Text completion** continues the raw prompt and does not render
   tools into it; the decoder still receives the catalog.

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

Tool names must be unique. Catalogs are limited to 128 function tools and
256 KiB of serialized JSON. Nested JSON Schema is passed through to your
decoder. Weave validates the catalog envelope; the decoder owns schema support
and grammar enforcement. The saved template must actually render `tools` to
make declarations visible to the model. This uses the documented
[Transformers tool-template interface](https://huggingface.co/docs/transformers/main/en/chat_extras).

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
Special XML token spellings are preserved in constrained responses; terminal
EOS is omitted from their displayed text. With no decoder and no tools, prompt
rendering, sampling and decoding retain their existing behavior.

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

Omitting `tools` or `decoder` inherits that loaded session's value. Explicit
`null` clears it; `tools: []` supplies an explicitly empty catalog. Optional
per-message fields override the defaults for that response and become the
session's current settings. A session started without these options defaults
to `None` for both, preserving older clients. For conversation requests with a
catalog, the worker calls `apply_chat_template(messages, tools=tools, ...)`.

## Verification

```bash
USE_TORCH=1 .venv/bin/python -m pytest tests/test_chat_decoders.py tests/test_checkpoint_chat.py -q
npx playwright test tests/browser/checkpointChat.spec.ts
```

The integration tests use real frozen model exports/checkpoints and an
allowlisted dummy decoder forcing one token, covering greedy and sampled output,
fresh response state, tool-template forwarding, bad masks, catalog validation,
session defaults and unchanged source files. XML-specific guarantees must also
be tested against your installed grammar and tokenizer, including all declared
tools, forbidden tool/argument names, escaped values, `<no_call/>`, EOS and token
budget exhaustion. A forced-token dummy is not an XML-grammar correctness test.
