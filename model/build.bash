docker build -t my-image .
# docker run my-image /script/to/run/tests
docker tag my-image aneequekhalid/dummy_test:v2

cat my_password.txt | sudo docker login --username aneequekhalid --password-stdin

docker push aneequekhalid/dummy_test:v2
