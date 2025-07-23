FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем requirements.txt
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Копируем ВСЕ файлы включая папку src
COPY . .

# Посмотрим структуру (для отладки)
RUN ls -la && ls -la src/

# Создаем директории
RUN mkdir -p models data logs

# Переменные окружения
ENV PYTHONPATH=/app

EXPOSE 8501

# Правильный путь к файлу в папке src
CMD ["python", "-m", "streamlit", "run", "src/streamlit_app_v2.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
