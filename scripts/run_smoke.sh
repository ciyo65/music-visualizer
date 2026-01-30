#!/usr/bin/env bash
set -e
PORT=${1:-8502}
streamlit run app.py --server.port "$PORT" --server.headless true
