"""E2E conftest — hits a live ihd-faiss-backend container over shdnetwork.

Routes stable across branches (no /api/* refactor here).
"""
import os
import httpx
import pytest

BASE_URL = os.getenv("E2E_BASE_URL", "http://ihd-faiss-backend")

ROUTES = {
    "health": os.getenv("E2E_ROUTE_HEALTH", "/api/health"),
    "projects": os.getenv("E2E_ROUTE_PROJECTS", "/api/projects"),
    # search path is templated: /api/projects/<id>/search
    "search_template": os.getenv("E2E_ROUTE_SEARCH", "/api/projects/{project_id}/search"),
}


@pytest.fixture(scope="session")
def base_url():
    return BASE_URL


@pytest.fixture(scope="session")
def routes():
    return ROUTES


@pytest.fixture(scope="session")
def client():
    headers = {"User-Agent": "ihd-faiss-e2e/1.0"}
    token = os.getenv("API_TOKEN", "").strip()
    if token:
        headers["X-API-Key"] = token
    c = httpx.Client(base_url=BASE_URL, headers=headers, timeout=60.0)
    yield c
    c.close()


@pytest.fixture(scope="session")
def healthy(client, routes):
    r = client.get(routes["health"])
    if r.status_code != 200:
        pytest.skip(f"service not healthy ({r.status_code} @ {routes['health']})")
    return True


@pytest.fixture(scope="session")
def first_project(client, routes, healthy):
    r = client.get(routes["projects"])
    if r.status_code != 200:
        pytest.skip(f"projects {r.status_code}")
    data = r.json()
    projects = data.get("projects") or {}
    if not projects:
        pytest.skip("no projects configured")
    pid, pdata = next(iter(projects.items()))
    sops = pdata.get("sops") or {}
    if not sops:
        pytest.skip(f"project {pid} has no SOPs")
    # pick first sop_key
    first_sop = next(iter(sops.values()))
    return {
        "project_id": pid,
        "sop_key": first_sop.get("sop_key"),
        "sop_title": next(iter(sops.keys())),
    }
