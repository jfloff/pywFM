# Docker image to run pywFM examples
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
    pip install --upgrade --no-cache-dir scipy \
                                         scikit-learn \
                                         pandas \
                                         ;\
    pip install --upgrade --no-cache-dir pywFM ;\
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
WORKDIR /home/pywfm

# start init script and bash right after
CMD /bin/bash
