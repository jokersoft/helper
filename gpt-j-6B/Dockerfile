FROM nvidia/cuda:11.8.0-base-ubuntu20.04
RUN apt update\
  && apt install -y python3 python3-pip wget git zstd curl\
  && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y nvidia-cuda-toolkit
RUN wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 0 -c https://the-eye.eu/public/AI/GPT-J-6B/step_383500_slim.tar.zstd\
  && tar -I zstd -xf step_383500_slim.tar.zstd\
  && rm step_383500_slim.tar.zstd
RUN git clone https://github.com/kingoflolz/mesh-transformer-jax.git
RUN pip3 install -r mesh-transformer-jax/requirements.txt
RUN pip3 install torch mesh-transformer-jax/ jax==0.2.12 jaxlib==0.1.68 -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN mkdir gpt-j-6B &&\
  curl https://raw.githubusercontent.com/jokersoft/helper/main/gpt-j-6B/config.json > gpt-j-6B/config.json
COPY converttotorch.py ./
RUN python3 converttotorch.py
RUN pip3 install fastapi pydantic uvicorn && pip3 install numpy --upgrade && pip3 install git+https://github.com/finetuneanon/transformers@gpt-j
RUN pip3 install protobuf==3.20.* && pip3 install numpy==1.23.4
COPY web.py ./
COPY model.py ./
CMD uvicorn web:app --port 8080 --host 0.0.0.0
