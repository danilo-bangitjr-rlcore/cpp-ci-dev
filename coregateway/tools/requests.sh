#!/bin/bash

# Needs xh
# `$ cargo install xh`
# Can replace xh for curl -X

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
get-db)
	xh GET localhost:8001/api/v1/coretelemetry/api/config/db
	;;
set-db)
	xh POST localhost:8001/api/v1/coretelemetry/api/config/db \
		drivername=postgresql+psycopg2 \
		username=postgres \
		password=password \
		ip=localhost \
		port:=5433 \
		db_name=postgres \
		schema=public
	;;
*)
	echo "Usage: $0 {my-desired-endpoint}"
	echo "  ..."
	exit 1
	;;
esac
echo
