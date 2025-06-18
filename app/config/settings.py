from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment-specific configuration.
    """

    # API Keys
    GOOGLE_API_KEY: str
    MISTRAL_API_KEY: str

    # Application Settings
    APP_NAME: str = "HealthPay API"
    APP_VERSION: str = "1.0.0"  # Full semantic version
    API_VERSION: str = "v1"  # API route version
    DEBUG: bool = False

    # AI Model Settings
    GEMINI_MODEL: str = "gemini-2.5-flash"
    MISTRAL_OCR_MODEL: str = "mistral-ocr-latest"

    # Processing Settings
    MAX_FILE_SIZE_MB: int = 50
    SUPPORTED_FILE_TYPES: list = ["application/pdf"]
    MAX_FILES_PER_REQUEST: int = 10

    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "detailed"  # "simple" or "detailed"

    # Performance Settings
    OCR_TIMEOUT_SECONDS: int = 60
    AI_PROCESSING_TIMEOUT_SECONDS: int = 120

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )


Config = Settings()
