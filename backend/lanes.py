"""Interfaces for explicit residual-lane wires; no tensor execution here."""

MAX_LANES = 32
MULTI_OUTPUT_LANES = frozenset(("lane_fork", "lane_split"))


def lane_count(settings):
    value = settings.get("lanes", 4)
    if type(value) is not int or not 1 <= value <= MAX_LANES:
        raise ValueError(f"Lane count must be an integer between 1 and {MAX_LANES}.")
    return value


def lane_ports(settings):
    try:
        return [f"lane{i}" for i in range(lane_count(settings))]
    except ValueError:
        return []  # Invalid settings are reported on the owning node.


def source_contract(report, key, handle=None):
    handle = handle or "out"
    ports = report.get("port_shapes", {}).get(key)
    if ports is not None:
        return ports.get(handle)
    return report["shapes"].get(key) if handle == "out" else None


def port_label(port):
    if port.startswith("lane") and port[4:].isdigit():
        return f"lane {int(port[4:]) + 1}"
    return port
