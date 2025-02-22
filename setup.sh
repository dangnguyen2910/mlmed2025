#!/bin/bash

mkdir -p data/heartbeat
curl -L -o ./data/heartbeat/heartbeat.zip\
  https://www.kaggle.com/api/v1/datasets/download/shayanfazeli/heartbeat

unzip data/heartbeat/heartbeat.zip -d data/heartbeat/

