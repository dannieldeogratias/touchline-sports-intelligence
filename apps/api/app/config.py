from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Touchline API"
    database_url: str = "sqlite+aiosqlite:///./touchline.db"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    media_root: Path = Path("D:/shayo/projects/footballanalysis/data/media")
    analysis_root: Path = Path("D:/shayo/projects/footballanalysis/data/analysis")
    max_upload_bytes: int = 1_500_000_000
    max_development_duration_ms: int = 610_000
    yolo_model_path: Path = Path("D:/shayo/projects/footballanalysis/models/football-detector-pilot.pt")
    ball_model_path: Path = Path("D:/shayo/projects/footballanalysis/models/football-ball-detector.pt")
    vision_python: Path = Path("D:/shayo/projects/footballanalysis/.venv-vision/Scripts/python.exe")
    vision_worker: Path = Path("D:/shayo/projects/footballanalysis/workers/vision/run_detection.py")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="TOUCHLINE_")

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
