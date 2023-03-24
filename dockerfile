FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install python3.7
RUN apt-get -y install python3-pip
RUN apt-get -y install python3-setuptools
WORKDIR /airbnb
COPY ./requirements.txt /airbnb/
COPY ./models /airbnb/models
COPY ./model_api /airbnb/model_api/
RUN pip3 install -r requirements.txt
CMD ["python3", "/airbnb/model_api/airbnb_api.py"]