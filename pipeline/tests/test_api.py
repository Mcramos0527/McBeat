import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health_check():
    assert client.get("/health").json() == {"status": "ok"}


def test_process_job_empty_body():
    resp = client.post("/process-job", json={})
    assert resp.status_code == 422


def test_process_job_blank_job_id():
    resp = client.post("/process-job", json={"job_id": "", "project_id": "proj-1"})
    assert resp.status_code == 422


@patch("api.run_pipeline")
def test_process_job_returns_202(mock_run):
    resp = client.post("/process-job",
                       json={"job_id": "job-abc", "project_id": "proj-xyz"})
    assert resp.status_code == 202


@patch("api.run_pipeline")
def test_process_job_calls_run_pipeline(mock_run):
    client.post("/process-job",
                json={"job_id": "job-abc", "project_id": "proj-xyz"})
    mock_run.assert_called_once_with("job-abc", "proj-xyz")


@patch("api.run_pipeline")
def test_process_job_response_body(mock_run):
    resp = client.post("/process-job",
                       json={"job_id": "job-abc", "project_id": "proj-xyz"})
    data = resp.json()
    assert data["job_id"] == "job-abc"
    assert data["status"] == "accepted"
