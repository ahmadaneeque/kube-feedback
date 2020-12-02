FROM debian:buster
RUN apt update && \
    apt install -y curl && \
    curl -LO https://storage.googleapis.com/kubernetes-release/release/`curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt`/bin/linux/amd64/kubectl && \
    chmod +x ./kubectl && \
    mv ./kubectl /usr/local/bin/kubectl
RUN apt install nano
COPY . /resources

RUN read -p "Press enter to continue"

