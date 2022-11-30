FROM python:3.7.3-slim

WORKDIR /src
COPY . /src

RUN apt-get update 
RUN ls -l
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "/src/app.py"]

