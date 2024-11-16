FROM jupyter/scipy-notebook:latest
WORKDIR /app
RUN pip install --upgrade pip

COPY # ПУТЬ К ФАЙЛУ НА КОМПЕ ПОТОМ ПРОБЕЛ ПОТОМ ПУТЬ К ФАЙЛУ В ДИРЕКТОРИИ IMAGE
ENTRYPOINT ["jupyter", "название файла", "--ip=0.0.0.0", "--port=8888", "--allow-root"]