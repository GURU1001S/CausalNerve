FROM python:3.11-slim

WORKDIR /causalnerve

# Install system dependencies including FFmpeg for animation export
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

# Copy source
COPY . .

# Install with development dependencies
RUN pip install -e ".[dev]"

# Verify installation and environment
RUN python -c "import causalnerve; print('CausalNerve is installed!')"
RUN python -c "import matplotlib.animation as anim; print('FFmpeg Available:', anim.writers.is_available('ffmpeg'))"

# Deterministic seed control
ENV PYTHONHASHSEED=42

# Default: run quick reproducibility check
CMD ["python", "reproduce.py", "--quick"]
