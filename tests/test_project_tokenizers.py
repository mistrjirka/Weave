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


def test_upload_bundle_preserves_special_tokens_and_readable_name(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from backend.tokenizers import TokenizerService

    monkeypatch.setattr(server, "DATA", tmp_path)
    service = TokenizerService(tmp_path / "tokenizers")
    monkeypatch.setattr(server, "tokenizers", service)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "</s>": 2}, unk_token="[UNK]"))
    files = [
        ("files", ("tokenizer.json", tokenizer.to_str(), "application/json")),
        ("files", ("tokenizer_config.json", json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "</s>", "unk_token": "[UNK]"}), "application/json")),
    ]
    client = TestClient(server.app)
    response = client.post("/api/tokenizers/upload", files=files)
    assert response.status_code == 200, response.text
    info = response.json()
    assert info["display_name"] == "tokenizer.json + 1 file"
    assert info["capacity"] == 3
    assert info["special_tokens"]["eos_token"] == {"token": "</s>", "id": 2}
    service.loaded.clear()
    preview = client.post("/api/tokenizers/preview", json={"id": info["id"], "text": "hello"})
    assert preview.status_code == 200
    assert preview.json()["ids"] == [1]
    metadata = json.loads((service.root / info["id"] / "studio_metadata.json").read_text())
    assert metadata["display_name"] == info["display_name"]
    duplicate = client.post("/api/tokenizers/upload", files=[files[0], files[0]])
    assert duplicate.status_code == 422
    assert "same name" in duplicate.json()["detail"]
    assert not list((tmp_path / "uploads").iterdir())
