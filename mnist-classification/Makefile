SHELL=/bin/bash

clean: clean-pyc clean-test clean-answer

base_image=ubuntu:18.04 #nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
python_ver=3.6.9
dir := ${CURDIR}

set-mnist-dataset:
	mkdir -p data/MNIST/processed
	mkdir -p data/MNIST/raw
	wget -P data/MNIST/raw/ https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/t10k-images-idx3-ubyte.gz
	wget -P data/MNIST/raw/ https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/t10k-labels-idx1-ubyte.gz
	wget -P data/MNIST/raw/ https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/train-images-idx3-ubyte.gz
	wget -P data/MNIST/raw/ https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/train-labels-idx1-ubyte.gz


#####  docker  #####
build-docker:
	docker build -f docker/Dockerfile -t mnist:v0 . --build-arg BASE_IMAGE=$(base_image) --build-arg PYTHON_VER=$(python_ver)

run-docker:
	docker run -i -t -d --shm-size=8G --init -v $(dir):/home/user/workspace/ --name mnist mnist:v0

exec-docker:
	docker exec -it mnist /bin/bash

rm-docker:
	docker stop mnist && docker rm mnist

#####  clean  #####

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f .coverage.*
	rm -rf .pytest_cache
	rm -rf .mypy_cache

clean-answer:
	rm -rf answer.json
