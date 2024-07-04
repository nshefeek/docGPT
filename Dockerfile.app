FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml *.lock README.md ./
RUN pip install pdm
RUN pdm install --no-editable


COPY /src .
COPY /scripts ./

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "docgpt.main:app", "--host", "0.0.0.0", "--port", "8000"]