FROM python:3.7-buster

WORKDIR /src
COPY . /src

RUN apt update && \
    apt install -y curl && \
    curl -LO https://storage.googleapis.com/kubernetes-release/release/`curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt`/bin/linux/amd64/kubectl && \
    chmod +x ./kubectl && \
    mv ./kubectl /usr/local/bin/kubectl

RUN ls -l
RUN pip install -r requirements.txt

RUN chmod +x ./build.bash
# RUN ./build.bash
# RUN python /src/training.py
# CMD ["python", "/src/app.py"]

CMD kubectl -n kube-feedback set image deployment/anomaly-model model=aneequekhalid/dummy_test:v2 --record=true
# CMD kubectl -n kube-feedback get pods

