#!/bin/bash

case "$1" in
  reward)
    xh GET localhost:8001/api/v1/coretelemetry/api/data/dep_mountain_car_continuous_wide metric==reward
    ;;
  get-path)
    xh GET localhost:8001/api/v1/coretelemetry/api/config/path
    ;;
  set-path)
    xh POST localhost:8001/api/v1/coretelemetry/api/config/path path==../config
    ;;
  *)
    echo "Usage: $0 {my-desired-endpoint}"
    echo "  ..."
    exit 1
    ;;
esac
