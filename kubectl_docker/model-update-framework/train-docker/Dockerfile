FROM aneequekhalid/python3.7-tf:latest

COPY training_model.py /app/training_model.py

WORKDIR /app
RUN ls -l

ENTRYPOINT ["python"]

CMD ["training_model.py"]
