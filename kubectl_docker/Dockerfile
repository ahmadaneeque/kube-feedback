FROM python:3.7-buster

WORKDIR /src
ADD . /src

RUN apt-get update 

RUN pip install -r /src/requirements.txt

EXPOSE 5000

CMD ["python", "/src/app.py"]

