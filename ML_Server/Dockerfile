FROM python:3.6.9

COPY ./sentence-transformers /sentence-transformers
COPY ./main.py /app/main.py

WORKDIR /sentence-transformers
ADD https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/paraphrase-distilroberta-base-v1.zip .
RUN unzip ./paraphrase-distilroberta-base-v1.zip -d ./paraphrase-distilroberta-base-v1 && rm ./paraphrase-distilroberta-base-v1.zip

RUN pip install -r requirements.txt
RUN pip install -e .

WORKDIR /app