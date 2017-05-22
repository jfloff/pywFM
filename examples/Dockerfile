# Docker image to run pywFM examples
FROM jfloff/alpine-python:2.7

# install needed packages
RUN echo "@community http://dl-cdn.alpinelinux.org/alpine/edge/community" >> /etc/apk/repositories \
    && echo "@testing http://dl-cdn.alpinelinux.org/alpine/edge/testing" >> /etc/apk/repositories \
    && apk add --update \
               py-numpy-f2py@community \
               py-numpy@community \
               py-numpy-dev@community \
    # split commands due some constrains bug
    && apk add --update \
               py-scipy@testing \
    && rm /var/cache/apk/*

# Due to dependencies scipy and scikit-learn have to be in different commands
# Also due to that, we can't have a requirements.txt file
RUN pip install --upgrade scipy \
    && pip install numpy \
    && pip install pandas \
    && pip install scikit-learn \
    && pip install pywFM

# clone repo and set envorinment variable to libfm PATH
RUN git clone https://github.com/srendle/libfm /home/libfm \
   && cd /home/libfm/ \
   # taking advantage of a bug to allow us to save model #ShameShame
   && git reset --hard 91f8504a15120ef6815d6e10cc7dee42eebaab0f \
   && make all
ENV LIBFM_PATH /home/libfm/bin/

# since we will be "always" mounting the volume, we can set this up
WORKDIR /home/pywfm

# start init script and bash right after
CMD /bin/bash
