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
- Load testing using Locust.

**Important:** This is an educational machine learning project, not a medical diagnostic system. Its predictions should not be used as a substitute for professional medical advice.

## 2. Dataset

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

## 3. Technologies

- **Python** – Main programming language.
- **TensorFlow / Keras** – Deep learning framework.
- **MobileNetV2** – Pretrained image classification backbone.
- **FastAPI** – REST API.
- **Streamlit** – Interactive web interface.
- **Docker and Docker Compose** – Containerization.
- **Locust** – Load testing.
- **Pandas, NumPy and Scikit-learn** – Data processing and evaluation.

## 4. Model Architecture

The project uses MobileNetV2 with transfer learning.

The model accepts RGB images of size 224 × 224 and predicts one of five diabetic retinopathy severity grades.

- Total parameters: 2,264,389
- Trainable parameters in the saved final model: 6,405
- Input size: 224 × 224 × 3
- Number of output classes: 5

Class weighting and early stopping were used during model development.

## 5. Model Evaluation

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

## 6. Application Architecture

The application has two main services:

**FastAPI**
- Loads the trained model.
- Provides a health-check endpoint.
- Accepts retinal images for prediction.
- Supports image uploads for retraining.
- Executes the retraining process and loads the resulting model.

**Streamlit**
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

Interactive API documentation is available at `/docs` when the API is running.

## 7. Model Retraining

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

The retraining workflow was successfully tested using 10 labelled images across two severity grades.

The resulting model and report are saved as:

- `models/diabetesvision_mobilenetv2_retrained.keras`
- `models/retraining_report.json`

**Retraining limitation:** The 10-image dataset was used to demonstrate that the pipeline works. Its small size is not sufficient to establish meaningful improvements in model accuracy or generalization.

## 8. Docker Deployment

The application is containerized using Docker Compose.

### Prerequisites

- Python 3.12
- Docker Desktop
- Docker Compose

### Run the application

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

### Access the application

- Streamlit frontend: http://localhost:8501
- FastAPI: http://localhost:8000
- API documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health

Stop the application:

```bash
docker compose down
```

The Dockerized application was successfully run and tested locally.

## 9. Load Testing with Locust

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

## 10. Project Structure

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
├── .dockerignore
├── .gitignore
├── compose.yaml
└── README.md
```

## 11. Demo Video

**Video demonstration:** [Add your video link here]

The demonstration should show:

1. The application running.
2. Uploading a retinal image.
3. Receiving a prediction and confidence score.
4. Uploading labelled images.
5. Triggering model retraining.
6. Confirming that the retrained model is loaded.
7. Running the Locust load tests.

## 12. Deployment Status and Limitations

- Local Docker deployment: completed and tested.
- API prediction: implemented and tested.
- Image upload and retraining: implemented and tested.
- Model evaluation: completed on a held-out test set.
- Locust load testing: completed for 5, 10 and 20 simulated users.
- Public cloud deployment: not yet completed.
- Multi-container scaling comparison: not yet completed.
- Image inference load testing: not yet completed.

## 13. Author

**GitHub:** [ksanthiana](https://github.com/ksanthiana)

**Repository:** https://github.com/ksanthiana/diabetesvision-mlops
