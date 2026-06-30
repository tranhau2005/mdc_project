from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "MDC Inference API")
    app_version: str = os.getenv("APP_VERSION", "0.1.0")
    model_config_path: str = os.getenv("MODEL_CONFIG_PATH", "configs/inference.yaml")


settings = Settings()
