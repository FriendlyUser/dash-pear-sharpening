#Retarded mechanism that uses hardcoded shit
FROM python:3.6.2
RUN mkdir -p /home/project/dash_app
WORKDIR /home/project/dash_app
COPY requirements.txt /home/project/dash_app
RUN pip install -r requirements.txt
COPY . /home/project/dash_app
# CMD gunicorn app:server