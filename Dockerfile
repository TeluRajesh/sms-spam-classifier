# Builds an image that trains the model and serves the REST API on port 5000.
# NOT built or run by the author of this repo's scaffolding: run `docker build` yourself and confirm it works.
FROM python:3.12-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY . .
# Train inside the image so the saved model always matches the installed scikit-learn version.
RUN python train.py
ENV HOST=0.0.0.0 PORT=5000
EXPOSE 5000
CMD ["python", "api.py"]
