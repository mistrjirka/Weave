"""Portable tuning identities and bounded search; no torch or web dependencies."""

import hashlib
import itertools
import json
import math


def graph_signature(project):
    def graph(value):
        return {
            "variables": value.get("variables", {}),
            "nodes": sorted(
                [
                    {
                        "id": node["id"],
                        "data": {
                            key: graph(item) if key == "body" else item
                            for key, item in node["data"].items()
                            if key not in ("annotations", "annotation", "description")
                        },
                    }
                    for node in value["nodes"]
                    if node["data"]["kind"] not in ("annotation", "variables")
                ],
                key=lambda item: item["id"],
            ),
            "edges": sorted(
                [
                    {
                        key: edge.get(key)
                        for key in ("source", "target", "sourceHandle", "targetHandle")
                    }
                    for edge in value["edges"]
                ],
                key=lambda item: json.dumps(item, sort_keys=True),
            ),
        }

    return hashlib.sha256(
        json.dumps(
            {"config": project["config"], "graph": graph(project["graph"])},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def allocate(total, speeds):
    """Positive integer shares with an exact, unchanged global microbatch size."""
    if (
        total < len(speeds)
        or not speeds
        or any(not math.isfinite(v) or v <= 0 for v in speeds)
    ):
        raise ValueError(
            "Need a positive speed per GPU and at least one sequence per GPU"
        )
    remaining = total - len(speeds)
    raw = [remaining * value / sum(speeds) for value in speeds]
    counts = [1 + math.floor(value) for value in raw]
    for index in sorted(range(len(speeds)), key=lambda i: raw[i] % 1, reverse=True)[
        : total - sum(counts)
    ]:
        counts[index] += 1
    return counts


def ddp_candidates(total, speeds, limit=8):
    estimate = allocate(total, speeds)
    candidates = [estimate, allocate(total, [1] * len(speeds))]
    for offset in sorted(set([1, max(1, total // 10)])):
        for source, target in itertools.permutations(range(len(speeds)), 2):
            candidate = estimate.copy()
            if candidate[source] > offset:
                candidate[source] -= offset
                candidate[target] += offset
                candidates.append(candidate)
    return list(dict.fromkeys(map(tuple, candidates)))[:limit]


def split_candidates(names, costs, devices, speeds, limit=8):
    """Contiguous stage placements, ordered by speed-proportional cost estimate."""
    if len(names) < len(devices):
        return []
    candidates = []
    # Bound enumeration even for graphs with thousands of root modules.
    count = len(devices)
    if math.comb(len(names) - 1, count - 1) <= 4096:
        cuts = itertools.combinations(range(1, len(names)), count - 1)
    else:
        cumulative = list(itertools.accumulate(costs))
        targets = list(itertools.accumulate(speeds))[:-1]
        initial = []
        for index, target in enumerate(targets):
            cut = min(
                range(
                    (initial[-1] + 1) if initial else 1,
                    len(names) - (count - index - 1) + 1,
                ),
                key=lambda n: abs(
                    cumulative[n - 1] / cumulative[-1] - target / sum(speeds)
                ),
            )
            initial.append(cut)
        cuts = [tuple(initial)]
    for cut in cuts:
        bounds = (0, *cut, len(names))
        loads = [
            sum(costs[bounds[i] : bounds[i + 1]]) / speeds[i] for i in range(count)
        ]
        placement = {
            names[j]: devices[i]
            for i in range(count)
            for j in range(bounds[i], bounds[i + 1])
        }
        ideal = sum(
            abs(
                sum(costs[bounds[i] : bounds[i + 1]]) / sum(costs)
                - speeds[i] / sum(speeds)
            )
            for i in range(count)
        )
        candidates.append((sum(loads), max(loads), placement, ideal))
    # Sequential split time is a sum, not the slowest-stage pipeline formula.
    candidates.sort(key=lambda row: (row[0], row[1]))
    if not candidates:
        return []
    initial = min(candidates, key=lambda row: row[3])
    return [initial[2], *[row[2] for row in candidates if row[2] != initial[2]]][:limit]
