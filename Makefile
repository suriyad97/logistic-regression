.PHONY: help install train evaluate deploy docker-build docker-run clean

help:
	@echo "Titanic Logistic Regression ML Pipeline"
	@echo ""
	@echo "Available targets:"
	@echo "  install         - Install Python dependencies"
	@echo "  train          - Train model locally"
	@echo "  evaluate       - Evaluate model"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run Docker container"
	@echo "  docker-compose - Run services with docker-compose"
	@echo "  deploy         - Deploy to Azure"
	@echo "  clean          - Clean up artifacts"

install:
	pip install -r requirements.txt

train:
	python src/train.py --data titanic.csv --output ./model

evaluate:
	python src/evaluate.py --model ./model/model.pkl --preprocessor ./model/preprocessor.pkl --data titanic.csv --output ./evaluation

docker-build:
	docker build -t titanic-logistic-regression:latest .

docker-run:
	docker run -p 5000:5000 -v $(PWD)/model:/model titanic-logistic-regression:latest

docker-compose:
	docker-compose up -d

docker-compose-down:
	docker-compose down

deploy:
	python scripts/submit_pipeline.py \
		--subscription-id $(AZURE_SUBSCRIPTION_ID) \
		--resource-group $(AZURE_RESOURCE_GROUP) \
		--workspace-name $(AZURE_ML_WORKSPACE) \
		--wait

clean:
	rm -rf model/ evaluation/ __pycache__ .pytest_cache *.pyc
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
