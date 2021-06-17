default: report.pdf packages
.PHONY : default


report.pdf: latex/report.tex figures/loss_epochs.jpg
	cd latex && pdflatex report.tex && bibtex report.aux && pdflatex report.tex && bibtex report.aux && pdflatex report.tex
	mv latex/report.pdf .


figures/loss_epochs.jpg: model_and_figures.py params_and_cli.py dataset/param_df.csv
	python model_and_figures.py


dataset/param_df.csv: 
	python generate_data.py --force


packages: env
	mkdir packages
	cd packages && git clone https://github.com/rhayes777/PyAutoFit
	cd packages && git clone https://github.com/Jammy2211/PyAutoArray
	cd packages && git clone https://github.com/Jammy2211/PyAutoGalaxy
	cd packages && git clone https://github.com/Jammy2211/PyAutoLens

	conda activate pyautolens && pip install -r packages/PyAutoFit/requirements.txt
	conda activate pyautolens && pip install -r packages/PyAutoArray/requirements.txt
	conda activate pyautolens && pip install -r packages/PyAutoGalaxy/requirements.txt
	conda activate pyautolens && pip install -r packages/PyAutoLens/requirements.txt

	conda-develop packages/PyAutoFit
	conda-develop packages/PyAutoArray
	conda-develop packages/PyAutoGalaxy
	conda-develop packages/PyAutoLens
.PHONY: packages


env:
	MY_ENV_DIR=$(shell conda info --base)/envs/$(pyautolens)
	ifneq ("$(wildcard $(MY_ENV_DIR))","")
		@echo "Found pyautolens environment"
	else
		@echo "Creating conda environment "pyautolens" ..."
		conda create -n pyautolens python==3.9
		conda activate pyautolens && pip install autoconf numpy pandas matplotlib tensorflow keras tqdm
.PHONY: env


clean:
	rm -rf dataset/* *.log *.swp
	rm -rf latex/*.log latex/*.aux latex/*.bbl latex/*.blg latex/*.pdf
.PHONY: clean


clean_all: clean
	rm -rf *.h5 *.pdf figures/*
.PHONY: clean_all
