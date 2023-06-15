#!/bin/bash

docker build . -t spode:1.0.1 \
--build-arg USER_ID=$(id -u) \
--build-arg GROUP_ID=$(id -g)
