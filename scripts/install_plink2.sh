#!/usr/bin/env bash

curl -o plink2_linux.zip "https://s3.amazonaws.com/plink2-assets/alpha4/plink2_linux_avx2_20230621.zip"
unzip plink2_linux.zip -d "gwasprs/bin/"
rm -f plink2_linux.zip
