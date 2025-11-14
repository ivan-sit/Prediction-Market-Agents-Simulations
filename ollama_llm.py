from typing import Dict, List, Optional, Union
from websocietysimulator.llm import LLMBase
import requests


class OllamaEmbeddings:
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        if response.status_code == 200:
            return response.json()['embedding']
        else:
            raise Exception(f"Ollama embedding error: {response.text}")


class OllamaLLM(LLMBase):
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        super().__init__(model)
        self.base_url = base_url
        self.embedding_model = OllamaEmbeddings(base_url=base_url)
        self._check_model_available()

    def _check_model_available(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                available = [m['name'] for m in models]
                if not any(self.model in m for m in available):
                    print(f"Warning: Model '{self.model}' not found")
                    print(f"Run: ollama pull {self.model}")
        except Exception as e:
            print(f"Warning: Could not check Ollama models: {e}")

    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 500,
                 stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        if n > 1:
            print("Warning: Ollama doesn't support n>1")

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model or self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens}
            }
        )

        if response.status_code == 200:
            return response.json()['message']['content']
        else:
            raise Exception(f"Ollama error: {response.text}")

    def get_embedding_model(self):
        return self.embedding_model
