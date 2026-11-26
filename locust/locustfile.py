
from locust import HttpUser, task, between


class DiabetesVisionUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def check_health(self):
        self.client.get("/health", name="Health Check")

    @task(1)
    def check_home(self):
        self.client.get("/", name="Home")
