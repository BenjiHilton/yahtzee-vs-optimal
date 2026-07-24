FROM python:3.11-slim

WORKDIR /app

# deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app + the precomputed expected-score table (ev_table.pkl)
COPY . .

# Render injects $PORT (default 10000) and health-checks that port; bind + EXPOSE
# the same port so its checks don't cycle the instance. The server reads $PORT.
ENV PORT=10000
EXPOSE 10000

CMD ["python", "-m", "webapp.server"]
