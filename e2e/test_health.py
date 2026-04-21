def test_health_returns_200(client, routes):
    r = client.get(routes["health"])
    assert r.status_code == 200


def test_health_reports_projects(client, routes):
    r = client.get(routes["health"])
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"
    assert "projects" in data
    assert isinstance(data["projects"], list)


def test_health_time_is_iso(client, routes):
    r = client.get(routes["health"])
    data = r.json()
    assert "time" in data and "T" in data["time"]


def test_health_is_fast(client, routes):
    r = client.get(routes["health"])
    assert r.elapsed.total_seconds() < 2.0, f"health slow: {r.elapsed}"
