# Docker image to pywFM development
FROM jfloff/alpine-python:2.7

# install needed packages
RUN set -ex ;\
    echo "@community http://dl-cdn.alpinelinux.org/alpine/v$ALPINE_VERSION/community" >> /etc/apk/repositories ;\
    apk add --no-cache --update libgfortran \
                                openblas-dev@community

# Due to dependencies scipy and scikit-learn have to be in different commands
# Also due to that, we can't have a requirements.txt file
RUN set -ex ;\
    pip install --upgrade --no-cache-dir numpy ;\
    pip install --upgrade --no-cache-dir scikit-learn \
                                         pandas \
                                         scipy \
                                         # for PyPi package management
                                         # How to publish to PyPi:
                                         # 1) bump setup.py for the new version <VERSION>
                                         # 2) `python setup.py bdist_wheel`
                                         # 3) `twine upload dist/pywFM-<VERSION>-py2-none-any.whl`
                                         wheel \
                                         twine \
                                         ;\
    # make sure nothing is on pip cache folder
    rm -rf ~/.cache/pip/

# clone repo and set envorinment variable to libfm PATH
RUN set -ex ;\
    git clone https://github.com/srendle/libfm /home/libfm ;\
    cd /home/libfm/ ;\
    # taking advantage of a bug to allow us to save model #ShameShame
    git reset --hard 91f8504a15120ef6815d6e10cc7dee42eebaab0f ;\
    make all
ENV LIBFM_PATH /home/libfm/bin/

# since we will be "always" mounting the volume, we can set this up
WORKDIR /home/pywFM

# install package in development mode at the begining
CMD pip install -e . && /bin/bash
