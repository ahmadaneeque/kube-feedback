FROM aneequekhalid/python3.7-ml:latest

COPY . /app

RUN pip install flask

WORKDIR /app

EXPOSE 5000

ENTRYPOINT ["python"]

CMD ["scoring.py"]
