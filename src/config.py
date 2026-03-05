from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_env: str = "development"
    debug: bool = True
    cors_origins: list[str] = ["http://localhost:3000"]

    # Database
    database_url: str = "postgresql+asyncpg://rag_user:rag_password@localhost:5432/rag_search"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Qdrant
    qdrant_url: str = ""  # Qdrant Cloud URL, e.g. https://xxx.cloud.qdrant.io:6333
    qdrant_api_key: str = ""  # Qdrant Cloud API key
    qdrant_host: str = "localhost"  # fallback for local dev
    qdrant_port: int = 6333
    qdrant_collection: str = "manual_chunks"

    # Embedding (AWS Bedrock)
    embedding_provider: str = "bedrock"  # "bedrock" | "openai"
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = 256
    embedding_max_retries: int = 3
    aws_bedrock_region: str = "ap-southeast-1"

    # Storage
    storage_backend: str = "local"  # "local", "s3", "gcs"
    storage_bucket: str = "rag-search-documents"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str = "us-east-1"

    # OpenAI
    openai_api_key: str = ""

    # Anthropic
    anthropic_api_key: str = ""

    # Stripe
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""

    # JWT
    jwt_secret_key: str = "change-this-to-a-random-secret"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30

    # Crawler
    crawler_data_dir: str = "./data/raw"
    crawler_rate_limit_seconds: float = 2.0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
