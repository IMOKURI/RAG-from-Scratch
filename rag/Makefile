.DEFAULT_GOAL := help

export
NOW = $(shell date '+%Y%m%d-%H%M%S')
ENV_VAL := "Hello world"

IMAGE_NAME = rag
IMAGE_TAG = 0.0.1


build-rag: ## Build container image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .


up-rag: ## Start RAG system
	docker run -it --rm --name rag-system -p 8501:8501 \
		--net rag-system \
		-v $(shell pwd):/code \
		-e OPENAI_BASE_URL="http://fastchat-api-server:8000/v1/" \
		-e OPENAI_API_KEY="EMPTY" \
		-e MODEL_NAME="gpt-3.5-turbo" \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python3 /code/app/main.py


down-rag: ## Stop RAG system
	docker stop rag-system || :


.PHONY: check-env
check-env: ## Check environment variables
	env | grep -E "(NOW|ENV_VAL)" || true

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[38;2;98;209;150m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
