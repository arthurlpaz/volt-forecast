"""Run the forecast API: python -m energycast.serving."""

from __future__ import annotations

import uvicorn

from energycast.config import get_settings
from energycast.serving.app import create_app


def main() -> None:
    settings = get_settings()
    serving = settings.base.serving
    uvicorn.run(create_app(settings), host=serving.host, port=serving.port)


if __name__ == "__main__":
    main()
