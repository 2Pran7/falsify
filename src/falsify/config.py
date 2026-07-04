"""Central config. Everything comes from environment variables (.env)."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    polygon_api_key: str = os.getenv("POLYGON_API_KEY", "")
    db_dsn: str = os.getenv(
        "DATABASE_URL",
        "postgresql://falsify:falsify_dev@localhost:5432/falsify",
    )
    # Free tier = 5 req/min; the Starter plan is unlimited.
    polygon_rpm: int = int(os.getenv("POLYGON_RPM", "5"))


settings = Settings()

if not settings.polygon_api_key:
    # Fail loudly at import time in scripts, not silently mid-run.
    import warnings

    warnings.warn("POLYGON_API_KEY is not set — ingest will fail.", stacklevel=1)
