"""Saved tokenizers can be reused across projects, including nested inputs."""

import json

from fastapi.testclient import TestClient
import backend.app as server


def test_saved_project_tokenizers_include_nested_and_legacy_bindings(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(server, "DATA", tmp_path)
    info = {
        "id": "a" * 20,
        "capacity": 4,
        "size": 4,
        "base_size": 3,
        "class": "WordLevel",
        "mask_id": 3,
        "special_tokens": {"mask": {"token": "[MASK]", "id": 3}},
    }
    other = {**info, "id": "b" * 20, "capacity": 8}

    def node(name, tokenizer):
        return {"id": name, "data": {"kind": "input", "tokenizer": tokenizer}}

    documents = [
        {
            "id": "nested",
            "name": "Nested model",
            "graph": {
                "nodes": [
                    node("tokens", info),
                    node("duplicate", info),
                    {
                        "id": "block",
                        "data": {
                            "kind": "container",
                            "body": {"nodes": [node("nested_tokens", other)]},
                        },
                    },
                ]
            },
        },
        {
            "id": "legacy",
            "name": "Older model",
            "tokenizer": info,
            "graph": {"nodes": []},
        },
        {"id": "empty", "name": "Empty", "graph": {"nodes": []}},
    ]
    with server.db() as conn:
        for document in documents:
            conn.execute(
                "INSERT INTO projects(id,name,document) VALUES(?,?,?)",
                (document["id"], document["name"], json.dumps(document)),
            )
    with TestClient(server.app) as client:
        response = client.get("/api/tokenizers/projects")
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 3
    assert {row["project_name"] for row in rows} == {"Nested model", "Older model"}
    nested = [row for row in rows if row["project_id"] == "nested"]
    assert [row["input_path"] for row in nested] == ["tokens", "block/nested_tokens"]
    assert nested[0]["tokenizer"] == info
    assert nested[1]["tokenizer"] == other
