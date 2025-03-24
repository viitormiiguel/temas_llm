
FROM python:3.9-slim

WORKDIR /app

COPY 'requirements.txt' .

RUN apt-get clean && apt-get -y update && apt-get install -y build-essential 

RUN pip install --upgrade pip setuptools

RUN pip install python-dotenv==1.0.1

RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8081

ENTRYPOINT [ "streamlit", "run", "main.py" ]