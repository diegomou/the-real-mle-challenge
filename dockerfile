FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install python3.7
RUN apt-get -y install python3-pip
RUN apt-get -y install python3-setuptools
WORKDIR /model_api
COPY ./models /model_api/models
COPY ./requirements.txt /model_api/
COPY ./model_api /model_api/
RUN pip3 install -r requirements.txt
CMD ["python3", "airbnb_api.py"]