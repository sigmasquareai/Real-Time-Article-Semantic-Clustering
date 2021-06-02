FROM python:3.6.9

RUN apt-get update
RUN apt-get install -y curl 

COPY ./app/requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt