"""Search endpoint e2e — hits real OpenAI embedding API.

Gated by RUN_EXPENSIVE_E2E because each call costs a few $0.0001
for the embedding. Cheap but still non-zero.
"""
import os
import pytest

SKIP_EXPENSIVE = os.getenv("RUN_EXPENSIVE_E2E", "").lower() not in ("1", "true", "yes")

pytestmark = pytest.mark.skipif(
    SKIP_EXPENSIVE,
    reason="search calls OpenAI embedding API; set RUN_EXPENSIVE_E2E=1 to enable",
)

QUERY = "洗手技術的基本步驟"


def _search_url(routes, project_id: str) -> str:
    return routes["search_template"].format(project_id=project_id)


def test_search_rejects_empty_query(client, routes, first_project):
    r = client.post(_search_url(routes, first_project["project_id"]), json={"query": ""})
    assert r.status_code == 400


def test_search_404_on_missing_project(client, routes, healthy):
    r = client.post(_search_url(routes, "__definitely_not_exists__"), json={"query": QUERY})
    assert r.status_code == 404


def test_search_returns_chunks(client, routes, first_project):
    r = client.post(
        _search_url(routes, first_project["project_id"]),
        json={"query": QUERY, "top_k": 3},
        timeout=60.0,
    )
    assert r.status_code == 200, f"body: {r.text[:500]}"
    data = r.json()
    chunks = data.get("chunks")
    assert isinstance(chunks, list), f"no chunks in: {list(data)}"
    assert len(chunks) > 0, "zero chunks returned"
    for c in chunks:
        assert "content" in c and c["content"]
        assert "sop_title" in c
        assert "sop_key" in c
        assert "score" in c


def test_search_respects_sop_keys_filter(client, routes, first_project):
    r = client.post(
        _search_url(routes, first_project["project_id"]),
        json={
            "query": QUERY,
            "top_k": 3,
            "sop_keys": [first_project["sop_key"]],
        },
        timeout=60.0,
    )
    assert r.status_code == 200
    data = r.json()
    for c in data.get("chunks", []):
        assert c["sop_key"] == first_project["sop_key"], (
            f"chunk from other SOP leaked: {c['sop_key']}"
        )


def test_search_timings_present(client, routes, first_project):
    r = client.post(
        _search_url(routes, first_project["project_id"]),
        json={"query": QUERY, "top_k": 2},
        timeout=60.0,
    )
    assert r.status_code == 200
    timings = r.json().get("timings") or {}
    assert "embed_ms" in timings
    assert "search_ms" in timings
