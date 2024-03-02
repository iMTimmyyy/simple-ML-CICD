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

	echo "\n## Confusion Matrix" >> report.md
	echo "![Confusion Matrix](./Results/confusion_matrix.png)" >> report.md

	cml comment create report.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push --force origin HEAD:update

hf-login:
	git pull origin update 
	git switch update
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HUGGINGFACE_TOKEN) --add-to-git-credential

push-hub:
	huggingface-cli upload iMTimmyyy/drug-classification ./App --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload iMTimmyyy/drug-classification ./Model /Model --repo-type=space --commit-message="Sync Model files"
	huggingface-cli upload iMTimmyyy/drug-classification ./Results /Metrics --repo-type=space --commit-message="Sync Results files"

deploy-hf: hf-login push-hub