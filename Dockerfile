FROM python:3.9.7

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "wsgi:app"]


