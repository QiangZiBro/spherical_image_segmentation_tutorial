ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN export http_proxy="" &&  export https_proxy="" && apt-get update && apt-get install -y curl ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
WORKDIR /pyigl
# Install libigl
RUN git clone --recursive https://github.com/libigl/libigl.git
RUN cd libigl \
    && mkdir external/nanogui/ext/glfw/include/GL \
    && wget --no-check-certificate -P external/nanogui/ext/glfw/include/GL http://www.opengl.org/registry/api/GL/glcorearb.h \
    && cd python \
    && mkdir build \
    && cd build \
    && cmake -DLIBIGL_WITH_NANOGUI=ON -DLIBIGL_USE_STATIC_LIBRARY=ON  -DCMAKE_CXX_COMPILER=g++-4.8 -DCMAKE_C_COMPILER=gcc-4.8 -DLIBIGL_WITH_EMBREE=OFF .. \
    && make -j 2 \
    && cd .. && mv nanogui nanogui.so

# wrapping up libigl core and helper in single module
RUN cp libigl/python/*.so /opt/conda/lib/ \
    && cp libigl/python/iglhelpers.py  /opt/conda/lib/
    
WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt
