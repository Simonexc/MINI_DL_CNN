install_requirements:
	pip install --upgrade pip
	pip install -r requirements.txt

docker_build:
	docker build -t dl-cnn .

docker_start:
	docker run -p 8888:8888 -v ./src:/workspace -v ./docker_setup:/root -v ./data:/data --gpus all dl-cnn
