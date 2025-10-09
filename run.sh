#!/usr/bin/env bash

# installer uv (si pas déjà)
python3 -m pip install uv

# synchroniser les dépendances
uv sync

# exécuter testImg puis main avec uv
uv run python testImg.py
uv run python main.py
