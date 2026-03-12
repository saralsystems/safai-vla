.PHONY: install lint test collect-data export-data train eval clean

install:
	pip install -e ".[all]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest envs/mujoco/test_env.py -v

test-policies:
	python -m policies.navigate --test --episodes 50
	python -m policies.position --test --episodes 50
	python -m policies.extract --test --episodes 50
	python -m policies.deposit --test --episodes 50

collect-data:
	python -m data.collect --task all --episodes 500 --output data/raw/

export-data:
	python -m data.export_lerobot --input data/raw/ --output data/lerobot/

push-data:
	python -m data.push_to_hub --dataset data/lerobot/ --repo saral-systems/sewer-vla-mujoco-v0

train:
	python -m training.finetune --dataset data/lerobot/ --output checkpoints/v0/

eval:
	python -m evaluation.sewerbench --checkpoint checkpoints/v0/ --episodes 100

clean:
	rm -rf data/raw/ data/lerobot/ checkpoints/ evaluation/results/
	find . -type d -name __pycache__ -exec rm -rf {} +

all: install lint test collect-data export-data train eval
