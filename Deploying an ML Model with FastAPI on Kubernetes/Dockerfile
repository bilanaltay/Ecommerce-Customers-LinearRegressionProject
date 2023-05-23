FROM tiangolo/uvicorn-gunicorn:python3.9

LABEL maintainer="Sebastian Ramirez <tiangolo@gmail.com>"

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]