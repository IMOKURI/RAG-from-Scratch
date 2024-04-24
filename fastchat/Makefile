.DEFAULT_GOAL := help

export
NOW = $(shell date '+%Y%m%d-%H%M%S')
ENV_VAL := "Hello world"

IMAGE_NAME = fastchat
IMAGE_TAG = 0.0.1


.PHONY: build-fastchat
build-fastchat: ## Build container image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .


up-fastchat-controller: ## Start FastChat controller.
	docker run -d --rm --name fastchat-controller -p 20000:20000 \
		--net rag-system \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python3 -m fastchat.serve.controller --host 0.0.0.0 --port 20000

up-fastchat-model-worker: ## Start FastChat model worker.
	docker run -d --rm --name fastchat-model-worker \
		--net rag-system \
		--gpus all --shm-size=32g \
		-v $(XDG_CACHE_HOME):/root/.cache \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python3 -m fastchat.serve.model_worker \
		--model-names gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002 \
		--model-path lmsys/vicuna-7b-v1.5 \
		--worker-address http://fastchat-model-worker:21000 \
		--controller-address http://fastchat-controller:20000 \
		--host 0.0.0.0 \
		--port 21000 \
		--num-gpus 1

up-fastchat-api-server: ## Start FastChat API server.
	docker run -d --rm --name fastchat-api-server -p 8000:8000 \
		--net rag-system \
		-e FASTCHAT_WORKER_API_TIMEOUT=300 \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python3 -m fastchat.serve.openai_api_server \
		--controller-address http://fastchat-controller:20000 \
		--host 0.0.0.0 \
		--port 8000

down-fastchat: ## Stop FastChat API Server.
	docker stop fastchat-api-server || :
	docker stop fastchat-model-worker || :
	docker stop fastchat-controller || :


.PHONY: check-env
check-env: ## Check environment variables
	env | grep -E "(NOW|ENV_VAL)" || true

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[38;2;98;209;150m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
