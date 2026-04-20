# One-command reproducible environment for the Tredence case study.
#
# Build:  docker build -t self-pruning-net .
# Run:    docker run --rm -v $(pwd)/outputs:/app/outputs self-pruning-net
#
# On a machine with GPU, add `--gpus all` to the run command.

FROM python:3.11-slim

WORKDIR /app

# System deps for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY README.md report.md ./

# Default: run the full experiment. Override with:
#   docker run ... self-pruning-net python -m src.sanity_check
CMD ["python", "-m", "src.main"]
