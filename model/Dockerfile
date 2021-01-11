FROM python:3.7-buster

WORKDIR /src
COPY . /src

RUN apt-get update 
RUN ls -l
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "/src/app.py"]

