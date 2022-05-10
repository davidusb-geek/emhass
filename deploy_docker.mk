image_ref := $(shell git describe --tags --exact-match 2> /dev/null || git rev-parse --verify --short HEAD)

project := emhass
repo := core

.PHONY: 

clean_deploy: deploy
	docker save -o emhass_${image_ref}.tar ${project}/${repo}:${image_ref}

deploy:
	docker build -t ${project}/${repo}:${image_ref} -f Dockerfile .
