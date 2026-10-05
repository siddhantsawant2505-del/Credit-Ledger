"""Unit tests for the FastAPI backend (server/api.py)."""

import unittest
import sys
from pathlib import Path
import httpx

server_dir = Path(__file__).resolve().parent
if str(server_dir) not in sys.path:
    sys.path.insert(0, str(server_dir))

from api import app


class TestAPI(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        transport = httpx.ASGITransport(app=app)
        self.client = httpx.AsyncClient(transport=transport, base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_health_endpoint(self):
        response = await self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("models_available", data)

    async def test_models_benchmark_endpoint(self):
        response = await self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)

    async def test_predict_endpoint(self):
        payload = {
            "age": 35.0,
            "employment": "ft",
            "residential": "owner_mortgage",
            "annualIncome": 85000.0,
            "requestedPrincipal": 20000.0,
            "loanTerm": 36.0,
            "dtiRatio": 25.0,
            "utilizationRate": 30.0,
            "priorDelinquencies": 0,
            "selectedModel": "logistic_regression",
        }
        response = await self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("probabilityOfDefault", data)
        self.assertIn("riskTier", data)
        self.assertIn("expectedLoss", data)
        self.assertIn("shapContributions", data)
        self.assertGreaterEqual(data["probabilityOfDefault"], 0.0)
        self.assertLessEqual(data["probabilityOfDefault"], 1.0)


if __name__ == "__main__":
    unittest.main()
