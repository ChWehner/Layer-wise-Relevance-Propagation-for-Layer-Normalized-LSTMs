FROM python:3.6

WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind","0.0.0.0:5000", "-w 2", "main:app"]



