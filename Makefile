.PHONY: setup data train evaluate explain app clean all

# Setup project
setup:
	pip install -r requirements.txt

# Generate synthetic data
data:
	python src/generate_data.py
	python src/database_manager.py

# Preprocess data
preprocess:
	python src/data_preprocessing.py

# Train models
train:
	python src/train_model.py

# Generate SHAP explanations
explain:
	python src/model_explainability.py

# Launch Streamlit app
app:
	streamlit run app/streamlit_app.py

# Clean generated files
clean:
	rm -rf data/*.csv data/*.pkl models/*.pkl models/*.json outputs/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run full pipeline
all: setup data train explain
	@echo "Pipeline completed! Run 'make app' to launch the Streamlit interface."

# Docker commands
docker-build:
	docker build -t vendor-invoice-intelligence .

docker-run:
	docker run -p 8501:8501 vendor-invoice-intelligence
