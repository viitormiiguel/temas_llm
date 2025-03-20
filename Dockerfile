
FROM python:3.9-slim

WORKDIR /app

COPY 'requirements.txt' .

RUN apt-get clean && apt-get -y update && apt-get install -y build-essential cmake libopenblas-dev liblapack-dev libopenblas-dev liblapack-dev libgl1 libglib2.0-0 libsm6 libxrender1 libxext6

RUN pip install --upgrade pip setuptools

RUN pip install python-dotenv==1.0.1

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8081

ENTRYPOINT [ "python3", "src/main.py" ]