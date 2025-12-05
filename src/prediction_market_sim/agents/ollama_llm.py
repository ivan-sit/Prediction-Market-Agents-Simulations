from typing import Dict, List, Optional, Union
from .websocietysimulator.llm import LLMBase
import requests
import json


class OllamaEmbeddings:
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.session = requests.Session()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        EMBEDDING_DIM = 768  # nomic-embed-text dimension

        try:
            if not text or not text.strip():
                print("Warning: Empty text provided to embed_query, returning zero vector")
                return [0.0] * EMBEDDING_DIM

            response = self.session.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()

                if 'embedding' not in data:
                    raise ValueError(f"Response missing 'embedding' key: {data.keys()}")

                embedding = data['embedding']

                if not embedding:
                    raise ValueError("Embedding is empty list")

                return embedding
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")

        except Exception as e:
            text_preview = text[:100] + "..." if len(text) > 100 else text
            print(f"Warning: Embedding failed (len={len(text)}): {e}")
            print(f"   Text preview: {text_preview}")
            print(f"   Returning zero vector to prevent crash")

            return [0.0] * EMBEDDING_DIM


class OllamaLLM(LLMBase):
    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434", num_ctx: int = 2048):
        super().__init__(model)
        self.base_url = base_url
        self.num_ctx = num_ctx
        self.embedding_model = OllamaEmbeddings(base_url=base_url)
        self.session = requests.Session()
        self._cache = {}
        self._max_cache_size = 1000
        self._check_model_available()

    def _check_model_available(self):
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
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

        cache_key = (
            model or self.model,
            json.dumps(messages, sort_keys=True),
            temperature,
            max_tokens
        )

        if cache_key in self._cache:
            return self._cache[cache_key]

        response = self.session.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model or self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": self.num_ctx
                }
            }
        )

        if response.status_code == 200:
            result = response.json()['message']['content']
            
            if len(self._cache) >= self._max_cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = result
            
            return result
        else:
            raise Exception(f"Ollama error: {response.text}")

    def get_embedding_model(self):
        return self.embedding_model
