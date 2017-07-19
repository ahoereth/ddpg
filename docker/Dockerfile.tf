FROM ubuntu:16.04

# MAINTAINER Craig Citro <craigcitro@google.com>

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python3-dev \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        openjdk-8-jdk \
        openjdk-8-jre-headless \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN pip3 --no-cache-dir install \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        pandas \
        && \
    python3 -m ipykernel.kernelspec

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY run_jupyter.sh /

# Set up Bazel.

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/etc/bazel.bazelrc
# Install the most recent bazel release.
ENV BAZEL_VERSION 0.5.0
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Download and build TensorFlow.

RUN git clone https://github.com/tensorflow/tensorflow.git && \
    cd tensorflow && \
    git checkout r1.2
WORKDIR /tensorflow

# TODO(craigcitro): Don't install the pip package, since it makes it
# more difficult to experiment with local changes. Instead, just add
# the built directory to the path.

RUN ln -s /usr/bin/python3 /usr/bin/python

ENV CI_BUILD_PYTHON python3

# --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \

RUN tensorflow/tools/ci_build/builds/configured CPU \
    bazel build \
        --config=opt --copt=-msse4.1 --copt=-mavx --copt=-mavx2 --copt=-mfma \
            tensorflow/tools/pip_package:build_pip_package && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip3 && \
    pip3 --no-cache-dir install --upgrade /tmp/pip3/tensorflow-*.whl && \
    rm -rf /tmp/pip3 && \
    rm -rf /root/.cache
# Clean up pip wheel and Bazel cache when done.

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

WORKDIR /root
CMD ["/bin/bash"]


###########################################################################
## PROJECT SPECIFIC
###########################################################################

RUN apt-add-repository -y ppa:george-edison55/cmake-3.x \
 && echo deb http://archive.ubuntu.com/ubuntu trusty-backports main restricted universe multiverse | tee /etc/apt/sources.list.d/trusty-backports.list \
 && apt-get update -qq

RUN apt-get install -y -qq curl wget git unzip python-opengl xvfb vim

# pachi-py deps
RUN apt-get install -y -qq cmake cmake-data libav-tools

# Box2D deps
RUN apt-get install -y -qq -t trusty-backports swig3.0 \
 && ln -s /usr/bin/swig3.0 /usr/bin/swig

RUN pip3 install -U pip setuptools gym[all]

# Manually build Box2D
# RUN mkdir -p /tmp \
#  && git clone https://github.com/pybox2d/pybox2d /tmp/pybox2d_dev \
#  && cd /tmp/pybox2d_dev \
#  && python3 setup.py clean \
#  && python3 setup.py build \
#  && python3 setup.py develop

# Torcs
RUN apt-get install -y -qq \
    xautomation \
    libglib2.0-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev  \
    freeglut3-dev \
    libplib-dev \
    libopenal-dev \
    libalut-dev \
    libxi-dev \
    libxmu-dev \
    libxrender-dev \
    libxrandr-dev \
    libpng12-dev
RUN git clone https://github.com/ugo-nama-kun/gym_torcs.git /tmp/gym_torcs \
 && cd /tmp/gym_torcs/vtorcs-RL-color \
 && ./configure \
 && make \
 && make install \
 && make datainstall

CMD ["/run_jupyter.sh", "--allow-root"]

