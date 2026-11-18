"""Project settings, read from environment variables or .env (see .env.example)."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
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


class DbSettings(BaseSettings):
    """PostgreSQL connection settings (unprefixed names, see .env.example)."""

    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_superuser: str = "postgres"
    postgres_superuser_password: SecretStr = SecretStr("")
    credit_db_name: str = "credit_dw"
    credit_db_tablespace_dir: str = ""
    credit_etl_user: str = "credit_etl"
    credit_etl_password: SecretStr = SecretStr("")
    credit_ro_user: str = "credit_ro"
    credit_ro_password: SecretStr = SecretStr("")

    def conninfo(self, role: str, database: str | None = None) -> dict:
        """psycopg connect() keyword arguments for role 'super', 'etl' or 'ro'."""
        user, password = {
            "super": (self.postgres_superuser, self.postgres_superuser_password),
            "etl": (self.credit_etl_user, self.credit_etl_password),
            "ro": (self.credit_ro_user, self.credit_ro_password),
        }[role]
        return {
            "host": self.postgres_host,
            "port": self.postgres_port,
            "user": user,
            "password": password.get_secret_value(),
            "dbname": database or self.credit_db_name,
            "connect_timeout": 5,
        }


@lru_cache
def get_db_settings() -> DbSettings:
    return DbSettings()
