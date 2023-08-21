FROM python:3.10 AS builder

WORKDIR /app

# add requirements and install without cache for reduced storage overhead (for local deployments)
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN python3 -m spacy download en_core_web_sm


# Stage 2: Create the final slim image
FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY server /app/server
COPY client /app/client

EXPOSE 5000

CMD ["python", "server/app.py"]