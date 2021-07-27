up:
	docker-compose up -d
	make env

down:
	docker-compose down

env:
	docker-compose exec spherical-env bash

build:
	docker build -t spherical . --network host \
			--build-arg http_proxy=${http_proxy}\
			--build-arg https_proxy=${https_proxy}

build_without_proxy:
	docker build -t spherical .

