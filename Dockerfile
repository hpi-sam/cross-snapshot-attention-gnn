# syntax=docker/dockerfile:1

FROM python:3.10.0

WORKDIR /app
ENV PYTHONPATH=${PYTHONPATH}:${PWD} 
COPY pyproject.toml /app 

RUN pip3 install --upgrade pip
RUN pip3 install torch==1.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install torch_geometric && \
    pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cpu.html

RUN pip3 install poetry
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev