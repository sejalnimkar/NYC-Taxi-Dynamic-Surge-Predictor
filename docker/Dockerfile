FROM apache/spark:4.0.2-scala2.13-java17-python3-r-ubuntu

USER root

RUN pip install --no-cache-dir \
    notebook \
    pandas \
    pyarrow

WORKDIR /home/jovyan/work