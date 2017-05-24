FROM tensorflow/tensorflow:1.2.0-rc0-py3

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
RUN mkdir -p /tmp \
 && git clone https://github.com/pybox2d/pybox2d /tmp/pybox2d_dev \
 && cd /tmp/pybox2d_dev \
 && python3 setup.py clean \
 && python3 setup.py build \
 && python3 setup.py develop

CMD ["/run_jupyter.sh", "--allow-root"]

