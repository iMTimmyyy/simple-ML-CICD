install:
	pip install -U pip 
	pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" >> report.md
	cat ./Results/metrics.txt >> report.md

	echo "## Confusion Matrix" >> report.md
	echo "![Confusion Matrix](./Results/confusion_matrix.png)" >> report.md

	cml comment create report.md