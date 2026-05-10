"""Tests for PATCH endpoints across all node types."""


def _seed(client, graph_id="edit_test"):
    client.post("/api/v1/graphs", json={"graph_id": graph_id})
    client.post(f"/api/v1/graphs/{graph_id}/memories", json={
        "mode": "structured",
        "semantic": [
            {"semantic_memory": "Paris is the capital of France", "tags": ["geo", "france"]},
        ],
        "procedural": [
            {"subgoal": "deploy app", "procedural_memory": "run docker-compose up"},
        ],
        "episodic": [[
            {"observation": "saw login", "action": "clicked login"},
        ]],
    })
    return graph_id


def test_patch_semantic_text_reembeds_and_updates_search(client):
    gid = _seed(client, "sem_text")
    resp = client.patch(f"/api/v1/graphs/{gid}/semantic/0", json={
        "text": "London is the capital of the UK",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["node"]["text"] == "London is the capital of the UK"

    # The new text shows up in search; the old one no longer matches.
    hits_new = client.get(f"/api/v1/graphs/{gid}/search?q=London&node_type=semantic").json()
    assert hits_new["count"] == 1
    hits_old = client.get(f"/api/v1/graphs/{gid}/search?q=Paris&node_type=semantic").json()
    assert hits_old["count"] == 0


def test_patch_semantic_tags_reconciles_attachments(client):
    gid = _seed(client, "sem_tags")
    resp = client.patch(f"/api/v1/graphs/{gid}/semantic/0", json={
        "tags": ["geo", "europe"],  # drop 'france', add 'europe'
    })
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert sorted(payload["node"]["tags"]) == ["europe", "geo"]

    # 'france' tag still exists as a node, but no longer attached to semantic 0.
    france = client.get(f"/api/v1/graphs/{gid}/search?q=france&node_type=tag").json()
    assert france["count"] == 1
    assert france["nodes"][0]["n_semantics"] == 0


def test_patch_semantic_credibility(client):
    gid = _seed(client, "sem_cred")
    resp = client.patch(f"/api/v1/graphs/{gid}/semantic/0", json={"credibility": 3})
    assert resp.status_code == 200
    assert resp.json()["node"]["credibility"] == 3


def test_patch_semantic_no_fields_is_400(client):
    gid = _seed(client, "sem_noop")
    resp = client.patch(f"/api/v1/graphs/{gid}/semantic/0", json={})
    assert resp.status_code == 400


def test_patch_procedural_text_and_return(client):
    gid = _seed(client, "proc_edit")
    resp = client.patch(f"/api/v1/graphs/{gid}/procedural/0", json={
        "text": "kubectl apply -f deploy.yaml",
        "return": 1.5,
    })
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["node"]["text"] == "kubectl apply -f deploy.yaml"
    assert payload["node"]["return"] == 1.5

    hits = client.get(f"/api/v1/graphs/{gid}/search?q=kubectl&node_type=procedural").json()
    assert hits["count"] == 1


def test_patch_tag_renames_and_propagates_to_semantic(client):
    gid = _seed(client, "tag_rename")
    # Find the 'geo' tag id.
    tags = client.get(f"/api/v1/graphs/{gid}/search?q=geo&node_type=tag").json()
    tag_id = tags["nodes"][0]["id"]

    resp = client.patch(f"/api/v1/graphs/{gid}/tag/{tag_id}", json={"tag": "geography"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["node"]["tag"] == "geography"

    # Semantic 0 now lists 'geography' instead of 'geo'.
    sem = client.get(f"/api/v1/graphs/{gid}/node/semantic/0").json()
    assert "geography" in sem["node"]["tags"]
    assert "geo" not in sem["node"]["tags"]


def test_patch_tag_duplicate_is_409(client):
    gid = _seed(client, "tag_dup")
    tags = client.get(f"/api/v1/graphs/{gid}/search?q=geo&node_type=tag").json()
    geo_id = tags["nodes"][0]["id"]
    # Try to rename 'geo' to 'france' which already exists.
    resp = client.patch(f"/api/v1/graphs/{gid}/tag/{geo_id}", json={"tag": "france"})
    assert resp.status_code == 409


def test_patch_tag_importance(client):
    gid = _seed(client, "tag_imp")
    tags = client.get(f"/api/v1/graphs/{gid}/search?q=geo&node_type=tag").json()
    tag_id = tags["nodes"][0]["id"]
    resp = client.patch(f"/api/v1/graphs/{gid}/tag/{tag_id}", json={"importance": 7})
    assert resp.status_code == 200
    assert resp.json()["node"]["importance"] == 7


def test_patch_subgoal_renames_and_propagates(client):
    gid = "sg_rename"
    client.post("/api/v1/graphs", json={"graph_id": gid})
    # Insert an episodic + procedural that share the same subgoal string so
    # the rename has somewhere to propagate.
    client.post(f"/api/v1/graphs/{gid}/memories", json={
        "mode": "structured",
        "episodic": [[
            {"observation": "ssh", "action": "docker up", "subgoal": "deploy app"},
        ]],
        "procedural": [
            {"subgoal": "deploy app", "procedural_memory": "run docker-compose up"},
        ],
    })

    subgoals = client.get(f"/api/v1/graphs/{gid}/search?q=deploy&node_type=subgoal").json()
    assert subgoals["count"] >= 1
    sg_id = subgoals["nodes"][0]["id"]

    resp = client.patch(f"/api/v1/graphs/{gid}/subgoal/{sg_id}", json={
        "subgoal": "ship release",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["node"]["subgoal"] == "ship release"

    proc = client.get(f"/api/v1/graphs/{gid}/node/procedural/0").json()
    assert "ship release" in proc["node"]["subgoals"]

    # The episodic that fed the procedural and shared the old subgoal string
    # should have been renamed too.
    epi = client.get(f"/api/v1/graphs/{gid}/node/episodic/0").json()
    assert epi["node"]["subgoal"] == "ship release"


def test_patch_episodic_observation_and_action(client):
    gid = _seed(client, "epi_edit")
    resp = client.patch(f"/api/v1/graphs/{gid}/episodic/0", json={
        "observation": "saw signup",
        "action": "filled form",
    })
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["node"]["observation"] == "saw signup"
    assert payload["node"]["action"] == "filled form"


def test_patch_unknown_node_404(client):
    gid = _seed(client, "missing_node")
    resp = client.patch(f"/api/v1/graphs/{gid}/semantic/9999", json={"is_active": False})
    assert resp.status_code == 404
