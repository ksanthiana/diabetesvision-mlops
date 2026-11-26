
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_predict_without_image():
    response = client.post("/predict")
    assert response.status_code == 422


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
