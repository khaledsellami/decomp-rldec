FROM --platform=linux/amd64 python:3.10-slim

COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt
RUN mkdir /service
COPY protos/ /protos/
WORKDIR /service
COPY . .

EXPOSE 50150
ENTRYPOINT [ "python", "main.py" ]