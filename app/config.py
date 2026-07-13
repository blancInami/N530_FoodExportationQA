from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # GatewayPlus
    gw_authorization: str = ""
    embedding_gateway_url: str = ""
    llm_gateway_url: str = ""

    # Embedding Server
    embedding_host: str = "10.166.57.22"
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

    # SQL Server 2025 (used when DB_TYPE=mssql)
    mssql_host: str = "127.0.0.1"
    mssql_port: int = 1433
    mssql_user: str = "sa"
    mssql_password: str = ""
    mssql_db: str = "fes"
    mssql_driver: str = "ODBC Driver 18 for SQL Server"
    mssql_trust_cert: bool = False

    # Database Backend  ("postgres" | "mssql")
    db_type: str = "postgres"
    # Schema name — "public" for PostgreSQL, "dbo" for SQL Server
    db_schema: str = "public"

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
        if self.db_type.lower() == "mssql":
            driver = self.mssql_driver.replace(" ", "+")
            trust = "&TrustServerCertificate=yes" if self.mssql_trust_cert else ""
            return (
                f"mssql+aioodbc://{self.mssql_user}:{self.mssql_password}"
                f"@{self.mssql_host}:{self.mssql_port}/{self.mssql_db}"
                f"?driver={driver}{trust}"
            )
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
