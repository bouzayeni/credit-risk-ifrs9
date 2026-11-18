"""Project settings, read from environment variables or .env (see .env.example)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", env_prefix="CREDIT_", extra="ignore"
    )

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")

    def resolve(self, path: Path) -> Path:
        """Relative paths are relative to the repo root, not the current directory."""
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def raw_path(self) -> Path:
        return self.resolve(self.raw_dir)

    @property
    def processed_path(self) -> Path:
        return self.resolve(self.processed_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
