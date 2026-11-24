"""Stable Python identifiers for generated source."""
import builtins
import keyword
import re
import unicodedata


MODULE_NAMES = {
    "forward",
    "training",
    "train",
    "eval",
    "parameters",
    "named_parameters",
    "buffers",
    "named_buffers",
    "children",
    "named_children",
    "modules",
    "named_modules",
    "state_dict",
    "load_state_dict",
    "to",
    "cpu",
    "cuda",
    "float",
    "double",
    "half",
    "bfloat16",
    "apply",
    "requires_grad_",
    "zero_grad",
    "get_parameter",
    "get_submodule",
    "register_parameter",
    "register_buffer",
    "add_module",
    "register_module",
    "compile",
    "type",
    "share_memory",
    "extra_repr",
    "set_extra_state",
    "get_extra_state",
}
RESERVED = (
    set(keyword.kwlist)
    | set(dir(builtins))
    | MODULE_NAMES
    | {
        "self",
        "torch",
        "nn",
        "F",
        "step",
        "steps",
        "shared_modules",
        "loss_targets",
        "loss_chunk_size",
        "supports_chunked_loss",
    }
)


def identifier(label, fallback="tensor"):
    text = unicodedata.normalize("NFKD", str(label)).encode("ascii", "ignore").decode()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:64]
    if not text:
        text = fallback
    if text[0].isdigit():
        text = "layer_" + text
    if text in RESERVED:
        text += "_tensor"
    return text


class Names:
    def __init__(self, reserved=()):
        self.used = set(reserved)

    def take(self, base):
        name, index = base, 2
        while name in self.used:
            name = f"{base}_{index}"
            index += 1
        self.used.add(name)
        return name
