FROM python:3.11-slim

ENV LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WAITFORIT_VERSION="v2.4.1"

WORKDIR /repo

RUN apt-get update \
    && apt-get install -y groff less curl \
    && pip install --upgrade pip \
    && curl -o /usr/local/bin/waitforit -sSL https://github.com/maxcnunes/waitforit/releases/download/$WAITFORIT_VERSION/waitforit-linux_amd64 \
    && chmod +x /usr/local/bin/waitforit

COPY ./netrc /root
COPY ./pip /root/.pip
COPY requirements-test.txt ./requirements.txt

RUN pip install --no-cache-dir --requirement requirements.txt

COPY . ./
