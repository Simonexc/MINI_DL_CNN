FROM pytorch/pytorch:latest

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

WORKDIR /workspace

EXPOSE 8888
CMD ["jupyter", "notebook", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
