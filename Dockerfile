# Docker image to pywFM development
FROM jfloff/alpine-python:2.7

# install needed packages
RUN apk add --update \
              py-numpy@testing \
              py-numpy-dev@testing \
              py-scipy@testing \
    && rm /var/cache/apk/*

# Due to dependencies scipy and scikit-learn have to be in different commands
# Also due to that, we can't have a requirements.txt file
RUN pip install numpy \
    && pip install scipy \
                   pandas \
    && pip install scikit-learn \
                   wheel

# since we will be "always" mounting the volume, we can set this up
WORKDIR /home/pywFM

# start init script and bash right after
CMD /bin/bash
