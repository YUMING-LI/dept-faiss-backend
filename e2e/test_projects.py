import pytest


def test_projects_endpoint_responds(client, routes, healthy):
    r = client.get(routes["projects"])
    assert r.status_code == 200


def test_projects_has_at_least_one_project(client, routes, healthy):
    r = client.get(routes["projects"])
    data = r.json()
    projects = data.get("projects") or {}
    assert projects, f"no projects configured: {data}"


def test_projects_payload_shape(client, routes, healthy):
    r = client.get(routes["projects"])
    data = r.json()
    projects = data.get("projects") or {}
    total_sops = 0
    for pid, pdata in projects.items():
        assert isinstance(pdata, dict), f"project {pid} not dict"
        sop_count = pdata.get("sop_count", 0)
        sops = pdata.get("sops") or {}
        total_sops += sop_count or len(sops)
        # Each SOP should carry a sop_key
        for title, info in sops.items():
            assert info.get("sop_key"), f"SOP {title} missing sop_key"
    assert total_sops > 0, "no SOPs across all projects"
