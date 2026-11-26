# DiabetesVision MLOps

An end-to-end image-based machine learning and MLOps project for classifying diabetic retinopathy severity from retinal fundus images.

## 1. Project Overview

DiabetesVision MLOps uses deep learning to classify retinal images into five diabetic retinopathy severity grades (0–4).

The project demonstrates an end-to-end machine learning workflow, including:

- Dataset acquisition and exploration.
- Image preprocessing and preparation.
- Transfer learning using MobileNetV2.
- Model evaluation using multiple classification metrics.
- Image prediction through a REST API.
- Image upload and model retraining.
- A Streamlit user interface.
- Docker-based application deployment.
- Cloud deployment using Render.
- Load testing using Locust.

**Important:** This is an educational machine learning project, not a medical diagnostic system. Its predictions should not be used as a substitute for professional medical advice.

## 2. Live Application

DiabetesVision is publicly deployed using Render.

- **Live application (Streamlit):** https://diabetesvision-frontend.onrender.com
- **Backend API (FastAPI):** https://diabetesvision-mlops.onrender.com
- **API health check:** https://diabetesvision-mlops.onrender.com/health
- **API documentation:** https://diabetesvision-mlops.onrender.com/docs

The frontend and backend are deployed as separate Docker-based web services.

The live application was tested successfully. An uploaded retinal image was processed, and the application returned a predicted severity grade, classification and confidence score.

**Hosting note:** The application currently uses Render's free instances. Free services may spin down after inactivity, resulting in slower initial responses. Their filesystem is ephemeral, so uploaded files and retrained models are not guaranteed to persist across restarts.

## 3. Dataset

The project uses the **APTOS 2019 Blindness Detection** dataset from Kaggle.

The dataset contains 3,662 labelled retinal images, with five diabetic retinopathy severity classes.

| Grade | Severity |
|---|---|
| 0 | No diabetic retinopathy |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative diabetic retinopathy |

The labelled dataset was divided using stratified sampling:

| Dataset split | Images |
|---|---:|
| Training | 2,929 |
| Validation | 366 |
| Testing | 367 |
| **Total** | **3,662** |

### Image preprocessing

- Images were loaded and checked for integrity.
- Images were converted to RGB.
- Images were resized to 224 × 224 pixels.
- MobileNetV2 preprocessing was applied.
- Stratified splitting was used to preserve class distribution.

## 4. Technologies

- **Python** – Main programming language.
- **TensorFlow / Keras** – Deep learning framework.
- **MobileNetV2** – Pretrained image classification backbone.
- **FastAPI** – REST API.
- **Streamlit** – Interactive web interface.
- **Docker and Docker Compose** – Containerization.
- **Render** – Cloud hosting and deployment.
- **Locust** – Load testing.
- **Pandas, NumPy and Scikit-learn** – Data processing and evaluation.

## 5. Model Architecture

The project uses MobileNetV2 with transfer learning.

The model accepts RGB images of size 224 × 224 and predicts one of five diabetic retinopathy severity grades.

- Total parameters: 2,264,389
- Trainable parameters in the saved final model: 6,405
- Input size: 224 × 224 × 3
- Number of output classes: 5

Class weighting and early stopping were used during model development.

## 6. Model Evaluation

The final model was evaluated on a held-out test set of 367 labelled images.

| Metric | Result |
|---|---:|
| Test accuracy | 75.75% |
| Macro precision | 61.24% |
| Macro recall | 64.88% |
| Macro F1-score | 61.10% |
| Weighted F1-score | 76.59% |
| Quadratic weighted kappa | 0.8337 |
| Test loss | 0.6155 |

These results represent the held-out test evaluation of the saved final model. They do not establish clinical performance.

The detailed evaluation results are available in `models/test_metrics.json`.

## 7. Application Architecture

The application has two main services.

### FastAPI Backend

- Loads the trained model.
- Provides a health-check endpoint.
- Accepts retinal images for prediction.
- Supports image uploads for retraining.
- Executes the retraining process and loads the resulting model.

### Streamlit Frontend

- Provides the user interface.
- Allows users to upload retinal images.
- Displays the predicted severity grade.
- Displays prediction confidence and class probabilities.
- Shows API connection status.

### API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | API and model health |
| POST | `/predict` | Predict from an uploaded image |
| POST | `/retrain/upload` | Upload labelled images |
| POST | `/retrain` | Trigger model retraining |

Interactive API documentation is available at `/docs`.

## 8. Model Retraining

The retraining workflow demonstrates how newly uploaded labelled images can be used to update the model.

The process is:

1. Upload labelled images.
2. Store images and labels in `data/uploaded/`.
3. Validate the uploaded images and labels.
4. Preprocess the images.
5. Split the uploaded data into training and validation sets.
6. Retrain using the existing MobileNetV2 model.
7. Save the retrained model.
8. Load the new model into the API.

The retraining workflow was successfully tested locally using 10 labelled images across two severity grades.

The resulting model and report are saved as:

- `models/diabetesvision_mobilenetv2_retrained.keras`
- `models/retraining_report.json`

**Retraining limitation:** The 10-image dataset was used to demonstrate that the pipeline works. Its small size is not sufficient to establish meaningful improvements in model accuracy or generalization.

The cloud prediction workflow has been tested successfully. Cloud-based retraining and persistent storage have not yet been verified.



## 9. Docker Deployment

The application is containerized using Docker Compose.

### Prerequisites

- Python 3.12
- Docker Desktop
- Docker Compose

### Run the application locally

Clone the repository:

```bash
git clone https://github.com/ksanthiana/diabetesvision-mlops.git
cd diabetesvision-mlops
```

Build the Docker images:

```bash
docker compose build
```

Start the services:

```bash
docker compose up -d
```

Check the running containers:

```bash
docker compose ps
```

### Access the local application

- Streamlit frontend: http://localhost:8501
- FastAPI: http://localhost:8000
- API documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health

Stop the application:

```bash
docker compose down
```

The Dockerized application was successfully run and tested locally.

## 10. Cloud Deployment with Render

The application was deployed to Render using the Dockerfiles in the GitHub repository.

Two separate web services were created:

| Service | Technology | Status |
|---|---|---|
| `diabetesvision-mlops` | FastAPI | Live |
| `diabetesvision-frontend` | Streamlit | Live |

### Deployment configuration

**Backend service**
- Runtime: Docker
- Dockerfile: `api/Dockerfile`
- Environment variable: `MODEL_PATH`
- Port: 8000

**Frontend service**
- Runtime: Docker
- Dockerfile: `frontend/Dockerfile`
- Environment variable: `API_URL`
- Port: 8501

The frontend connects to the deployed backend through its public API URL.

### Cloud deployment verification

The following were verified:

- Backend service successfully deployed.
- API health endpoint returned a running status.
- Model loaded successfully in the cloud.
- Frontend service successfully deployed.
- Frontend communicated with the backend.
- An uploaded retinal image successfully received a prediction.

**Public application:** https://diabetesvision-frontend.onrender.com

## 11. Load Testing with Locust

Locust was used to simulate concurrent users accessing the API.

The tests checked the `/health` and `/` endpoints.

### Test results

| Simulated users | Total requests | Failures | Average response time | 95th percentile |
|---:|---:|---:|---:|---:|
| 5 | 145 | 0 | 5 ms | 9 ms |
| 10 | 283 | 0 | 5 ms | 9 ms |
| 20 | 582 | 0 | 6 ms | 10 ms |

All three tests completed with zero failed requests.

### Run a load test

Install Locust:

```bash
python -m pip install locust
```

Run the test with 5 users:

```bash
python -m locust -f locust/locustfile.py --host http://127.0.0.1:8000 --users 5 --spawn-rate 1 --run-time 1m --headless --csv=outputs/locust/test_5_users
```

Equivalent test runs were performed with 10 and 20 simulated users.

The generated CSV reports are stored in `outputs/locust/`.

**Testing limitation:** These results measure the API home and health endpoints, not image inference. The tests were performed locally with the existing Docker setup; they do not compare different Docker container counts.

## Automated Testing

DiabetesVision includes automated API tests using **Pytest** and FastAPI's `TestClient`.

### Tests implemented

| Test | Description | Expected result |
|---|---|---|
| Root endpoint | Checks whether the API responds | HTTP 200 |
| Health endpoint | Checks API health information | HTTP 200 |
| Prediction validation | Checks that requests without an uploaded image are rejected | HTTP 422 |

### Run the tests

Install the dependencies:

```bash
pip install -r api/requirements.txt
pip install httpx
```

Run the test suite:

```bash
python -m pytest tests/ -v
```

### Latest test results

- Total tests: 3
- Passed: 3
- Failed: 0

All three automated API tests passed successfully on November 26, 2026.


## API Health Monitoring

DiabetesVision includes a Python-based monitoring script that periodically checks the availability and health of the deployed FastAPI service.

### Monitoring features

- Checks the deployed API health endpoint.
- Records API status.
- Checks whether the ML model is loaded.
- Measures API response time.
- Records timestamps and errors.
- Saves monitoring results in a CSV file.

### Run the monitoring script

```bash
python monitoring/health_monitor.py
```

The script checks the API every 60 seconds. Press `Ctrl + C` to stop monitoring.

### Monitoring results

A monitoring session on November 26, 2026 produced the following results:

| Check | API status | Model loaded | Response time |
|---|---|---|---:|
| 1 | Healthy | Yes | 54.53 seconds |
| 2 | Offline | No response | Failed |
| 3 | Healthy | Yes | 3.61 seconds |

The initial response was slow, consistent with a possible Render free-instance cold start. One request also failed because the remote connection closed unexpectedly.

The results are recorded in:

`monitoring/health_logs.csv`

**Note:** These results represent a short monitoring session, not a long-term uptime measurement.


## 12. Project Structure

```text
diabetesvision-mlops/
│
├── api/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── uploaded/
│
├── frontend/
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.txt
│
├── locust/
│   └── locustfile.py
│
├── models/
│   ├── diabetesvision_mobilenetv2_final.keras
│   ├── diabetesvision_mobilenetv2_retrained.keras
│   ├── retraining_report.json
│   └── test_metrics.json
│
├── monitoring/
│   ├── health_monitor.py
│   └── health_logs.csv
│
├── notebook/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training_mobilenetv2.ipynb
│
├── outputs/
│   └── locust/
│
├── src/
│   └── retraining.py
│
├── tests/
│   └── test_api.py
|
├── .dockerignore
├── .gitignore
├── compose.yaml
└── README.md
```

## 13. Demo Video

**Video demonstration:** [Add your video link here]

The demonstration should show:

1. The application running.
2. Accessing the deployed application.
3. Uploading a retinal image.
4. Receiving a prediction and confidence score.
5. Uploading labelled images.
6. Triggering model retraining locally.
7. Confirming that the retrained model is loaded.
8. Running the Locust load tests.

## 14. Deployment Status and Limitations

| Component | Status |
|---|---|
| Dataset exploration and preprocessing | Completed |
| MobileNetV2 model training | Completed |
| Model evaluation | Completed |
| FastAPI prediction | Tested locally and in the cloud |
| Streamlit interface | Tested locally and in the cloud |
| Local Docker deployment | Completed and tested |
| Local image upload and retraining | Completed and tested |
| Public cloud deployment | Completed on Render |
| Cloud image prediction | Tested successfully |
| Cloud retraining and persistent storage | Not yet verified |
| Locust load testing | Completed for 5, 10 and 20 simulated users |
| Multi-container scaling comparison | Not yet completed |
| Image inference load testing | Not yet completed |

## 15. Author

**GitHub:** https://github.com/ksanthiana

**Repository:** https://github.com/ksanthiana/diabetesvision-mlops

**Live Application:** https://diabetesvision-frontend.onrender.com