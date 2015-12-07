#!/bin/sh

rm -rf pywFM/libfm
git clone https://github.com/srendle/libfm pywFM/libfm
cd pywFM/libfm
make
