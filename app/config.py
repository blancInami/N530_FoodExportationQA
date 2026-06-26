from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # GatewayPlus
    gw_authorization: str = ""
    embedding_gateway_url: str = ""
    llm_gateway_url: str = ""

    # Embedding Server
    embedding_host: str = "10.166.57.21"
    embedding_port: int = 40003
    embedding_model: str = "intfloat/multilingual-e5-large-instruct"
    embedding_dim: int = 1024

    # LLM Server
    llm_host: str = "10.166.57.22"
    llm_port: int = 40036
    llm_model: str = "gpt-oss-20b"

    # PostgreSQL
    pg_host: str = "127.0.0.1"
    pg_port: int = 5432
    pg_user: str = "postgres"
    pg_password: str = "postgres"
    pg_db: str = "fes"

    # Retrieval Parameters
    similarity_threshold: float = 0.75
    top_k: int = 30
    top_n: int = 5
    # Number of responsible units to return (based on classification rank)
    intent_top_n: int = 1
    # "independent": each track gets its own top_n quota (default)
    # "compete":     both tracks share a single top_n quota sorted by distance
    retrieval_merge_mode: str = "independent"

    # Chunking Parameters
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Logging
    log_level: str = "INFO"

    @property
    def use_gateway(self) -> bool:
        """True when GW_Authorization is configured — activates gateway URL and headers."""
        return bool(self.gw_authorization)

    @property
    def gw_headers(self) -> dict[str, str]:
        """Extra headers required by the GatewayPlus proxy. Empty dict when not in gateway mode."""
        if self.use_gateway:
            return {
                "GW_Authorization": self.gw_authorization,
                "Content-Type": "application/json",
            }
        return {}

    @property
    def embedding_url(self) -> str:
        if self.use_gateway and self.embedding_gateway_url:
            return self.embedding_gateway_url
        return f"http://{self.embedding_host}:{self.embedding_port}"

    @property
    def llm_url(self) -> str:
        if self.use_gateway and self.llm_gateway_url:
            return self.llm_gateway_url
        return f"http://{self.llm_host}:{self.llm_port}"

    @property
    def dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
