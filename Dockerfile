FROM python:3.11-slim

WORKDIR /app

# deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app + the precomputed expected-score table (ev_table.pkl)
COPY . .

# the server binds 0.0.0.0:$PORT (defaults to 8000)
ENV PORT=8000
EXPOSE 8000

CMD ["python", "-m", "webapp.server"]
