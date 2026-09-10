FROM node:20-alpine AS frontend-build
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .
COPY --from=frontend-build /frontend/build ./static
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
