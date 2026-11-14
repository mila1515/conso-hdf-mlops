FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PROJECT_ROOT=/app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Versions figées (requirements.lock) : l'image est reproductible
COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY pyproject.toml .
COPY src ./src
COPY dashboard ./dashboard
RUN pip install --no-cache-dir --no-deps -e .

# Utilisateur non root
RUN useradd --create-home --uid 1000 app && mkdir -p data artifacts reports evidently_workspace && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s   CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1
CMD ["uvicorn", "conso_hdf.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
