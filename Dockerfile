FROM tensorflow/tensorflow:2.12.0-gpu

WORKDIR /app

COPY . /app

# Update pip and install requirements
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "server.py"]