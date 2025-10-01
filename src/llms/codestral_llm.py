# clinical_agent/src/llms/codestral_llm.py
import httpx
import google.auth
from google.auth.transport.requests import Request
from typing import Optional, List, Any

class CodeStralClient:
    """
    Minimal client for Mistral CodeStral on Vertex AI Publisher Models.
    - Uses /predict with {instances:[{messages:[...] }], parameters:{...}}
    - Your project must have the publisher model enabled
    """

    def __init__(
        self,
        project_id: str = "clinical-copilot",   # YOUR project (typo fixed)
        region: str = "us-central1",
        model_name: str = "mistralai/codestral-2501@001",
        temperature: float = 0.3,
        max_output_tokens: int = 512,
        timeout_s: Optional[float] = None,
    ):
        self.project_id = project_id
        self.region = region
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s

    def _token(self) -> str:
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(Request())
        return creds.token

    def _url(self) -> str:
        # Publisher model predict endpoint
        return (
            f"https://{self.region}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{self.region}/publishers/mistralai/models/{self.model_name}:predict"
        )

    def generate(self, prompts: List[str]) -> List[str]:
        headers = {
            "Authorization": f"Bearer {self._token()}",
            "Content-Type": "application/json"
        }
        instances = [{"messages": [{"role": "user", "content": p}]} for p in prompts]
        payload = {
            "instances": instances,
            "parameters": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens
            }
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(self._url(), headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            outs: List[str] = []
            # Each prediction → candidates → content
            for pred in data.get("predictions", []):
                cand = (pred.get("candidates") or [{}])[0]
                outs.append((cand.get("content") or "").strip())
            return outs
