from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str
    pinecone_api_key: str = ""
    pinecone_index_name: str = "travel-agency"
    human_support_webhook: str = ""
    admin_token: str = "change_me"
    log_level: str = "INFO"
    allowed_origins: str = "http://localhost:5173"
    index_dir: str = "./data/index"
    docs_dir: str = "./docs"
    db_path: str = "./data/conversations.db"

    # RAG settings
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 20
    retrieval_top_n: int = 5
    similarity_threshold: float = 0.35
    confidence_threshold: float = 0.6

    # LLM settings
    gemini_model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"
    llm_temperature: float = 0.2
    llm_timeout: float = 10.0


settings = Settings()
