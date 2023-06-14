FROM python:3.9
MAINTAINER Yueh-Hua Tu<yuehhua.tu@ailabs.tw>

# basic packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        curl \
        htop \
        locales \
        tree \
        tzdata \
        vim \
        gcc \
        git

# time zone and languages
ENV TZ=Asia/Taipei \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

RUN locale-gen en_US.UTF-8 \
 && echo $TZ | tee /etc/timezone \
 && dpkg-reconfigure --frontend noninteractive tzdata

WORKDIR /gwasprs

COPY . /gwasprs/

RUN pip install virtualenv \
 && virtualenv venv \
 && source venv/bin/activate \
 && python --version ; pip --version \
 && pip install -r requirements.txt
