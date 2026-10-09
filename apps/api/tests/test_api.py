def create_org(client, name: str, slug: str) -> dict:
    response = client.post("/api/v1/organizations", json={"name": name, "slug": slug})
    assert response.status_code == 201
    return response.json()


def create_match(client, organization_id: str) -> dict:
    response = client.post(
        "/api/v1/matches",
        headers={"X-Organization-ID": organization_id},
        json={"home_team": "KMC FC", "away_team": "Coastal Union", "competition": "Development fixture"},
    )
    assert response.status_code == 201
    return response.json()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_organization_can_be_loaded_by_slug(client):
    organization = create_org(client, "Touchline Development", "touchline-development")
    response = client.get("/api/v1/organizations/by-slug/touchline-development")
    assert response.status_code == 200
    assert response.json()["id"] == organization["id"]


def test_rejected_events_are_hidden_by_default(client):
    organization = create_org(client, "Archive Test", "archive-test")
    match = create_match(client, organization["id"])
    headers = {"X-Organization-ID": organization["id"]}
    created = client.post(
        f"/api/v1/matches/{match['id']}/events",
        headers=headers,
        json={"timestamp_ms": 1000, "event_type": "PASS"},
    ).json()
    client.patch(
        f"/api/v1/events/{created['id']}",
        headers=headers,
        json={"verification_status": "REJECTED"},
    )

    assert client.get(f"/api/v1/matches/{match['id']}/events", headers=headers).json() == []
    archived = client.get(
        f"/api/v1/matches/{match['id']}/events?include_rejected=true", headers=headers
    ).json()
    assert len(archived) == 1


def test_match_isolation_between_organizations(client):
    first = create_org(client, "KMC Analysis", "kmc-analysis")
    second = create_org(client, "Coastal Analysis", "coastal-analysis")
    match = create_match(client, first["id"])

    own_response = client.get(
        f"/api/v1/matches/{match['id']}", headers={"X-Organization-ID": first["id"]}
    )
    other_response = client.get(
        f"/api/v1/matches/{match['id']}", headers={"X-Organization-ID": second["id"]}
    )

    assert own_response.status_code == 200
    assert other_response.status_code == 404


def test_match_team_names_can_be_customised(client):
    organization = create_org(client, "Custom Teams", "custom-teams")
    match = create_match(client, organization["id"])
    response = client.patch(
        f"/api/v1/matches/{match['id']}",
        headers={"X-Organization-ID": organization["id"]},
        json={"home_team": "Liverpool", "away_team": "Newcastle United"},
    )
    assert response.status_code == 200
    assert response.json()["home_team"] == "Liverpool"
    assert response.json()["away_team"] == "Newcastle United"


def test_event_update_preserves_revision_history(client):
    organization = create_org(client, "Touchline Lab", "touchline-lab")
    match = create_match(client, organization["id"])
    headers = {"X-Organization-ID": organization["id"]}

    created = client.post(
        f"/api/v1/matches/{match['id']}/events",
        headers=headers,
        json={
            "timestamp_ms": 198000,
            "event_type": "SHOT",
            "player_name": "F. Musa",
            "source": "AI_SUGGESTION",
            "verification_status": "UNVERIFIED",
            "note": "Candidate shot",
        },
    )
    assert created.status_code == 201
    event = created.json()

    updated = client.patch(
        f"/api/v1/events/{event['id']}",
        headers=headers,
        json={
            "player_name": "A. Salum",
            "source": "AI_CORRECTED",
            "verification_status": "CORRECTED",
            "note": "Analyst corrected the shooter",
            "changed_by": "analyst-01",
        },
    )
    assert updated.status_code == 200
    assert updated.json()["revision_number"] == 2

    revisions = client.get(f"/api/v1/events/{event['id']}/revisions", headers=headers)
    assert revisions.status_code == 200
    history = revisions.json()
    assert [item["revision_number"] for item in history] == [1, 2]
    assert history[0]["player_name"] == "F. Musa"
    assert history[0]["source"] == "AI_SUGGESTION"
    assert history[1]["player_name"] == "A. Salum"
    assert history[1]["changed_by"] == "analyst-01"


def test_video_metadata_records_rights_and_segment_duration(client):
    organization = create_org(client, "Development Lab", "development-lab")
    match = create_match(client, organization["id"])
    response = client.post(
        f"/api/v1/matches/{match['id']}/videos",
        headers={"X-Organization-ID": organization["id"]},
        json={
            "filename": "development-segment.mp4",
            "content_type": "video/mp4",
            "duration_ms": 600000,
            "source_start_ms": 1200000,
            "rights_status": "development_only",
        },
    )
    assert response.status_code == 201
    assert response.json()["duration_ms"] == 600000
    assert response.json()["rights_status"] == "development_only"


def test_video_upload_and_mock_detection_job(client, tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "media_root", tmp_path)
    organization = create_org(client, "Vision Lab", "vision-lab")
    match = create_match(client, organization["id"])
    headers = {"X-Organization-ID": organization["id"]}
    fake_mp4 = b"\x00\x00\x00\x18ftypmp42" + (b"0" * 128)
    upload = client.post(
        f"/api/v1/matches/{match['id']}/media",
        headers=headers,
        data={"duration_ms": "600000", "source_start_ms": "0"},
        files={"file": ("segment.mp4", fake_mp4, "video/mp4")},
    )
    assert upload.status_code == 201
    video = upload.json()
    assert (tmp_path / video["storage_key"]).exists()
    assert len(video["checksum_sha256"]) == 64

    created = client.post(
        f"/api/v1/matches/{match['id']}/jobs",
        headers=headers,
        json={"video_id": video["id"], "mode": "MOCK"},
    )
    assert created.status_code == 202
    job = client.get(f"/api/v1/jobs/{created.json()['id']}", headers=headers)
    assert job.status_code == 200
    assert job.json()["status"] == "SUCCEEDED"
    assert job.json()["progress"] == 100


def test_upload_rejects_segments_over_ten_minutes(client, tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "media_root", tmp_path)
    organization = create_org(client, "Duration Lab", "duration-lab")
    match = create_match(client, organization["id"])
    response = client.post(
        f"/api/v1/matches/{match['id']}/media",
        headers={"X-Organization-ID": organization["id"]},
        data={"duration_ms": "700000"},
        files={"file": ("long.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4")},
    )
    assert response.status_code == 422
