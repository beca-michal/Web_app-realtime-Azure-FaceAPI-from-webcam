FROM python:3.8

WORKDIR /app

COPY requirements.txt /app
RUN pip install --upgrade pip setuptools wheel
RUN pip install numpy
RUN pip install -r requirements.txt

#COPY ./app.py /app
COPY . /app

ENV FLASK_APP app.py

ENTRYPOINT [ "flask" ] 
CMD ["run", "--host", "0.0.0.0"]