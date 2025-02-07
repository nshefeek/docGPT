FROM python:3.12

WORKDIR /app

COPY pyproject.toml *.lock README.md ./
COPY /src .
COPY /scripts ./

RUN pip install uv
RUN uv venv .venv
RUN uv sync --no-group 

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "docgpt.main:app", "--host", "0.0.0.0", "--port", "8000"]