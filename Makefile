default: report.pdf
.PHONY : default


# the report depends on the tex file and the images
report.pdf: latex/report.tex figures/loss_epochs.jpg
	cd latex && pdflatex report.tex && bibtex report.aux && pdflatex report.tex && bibtex report.aux && pdflatex report.tex
	mv latex/report.pdf .


# the images depend on the data generated (generated with the csv file)
figures/loss_epochs.jpg: model_and_figures.py params_and_cli.py dataset/param_df.csv
	python model_and_figures.py


# generate the data giles if needed
dataset/param_df.csv: 
	python generate_data.py --force


# clone all the required modules to ./packages
packages:
	mkdir packages
	cd packages && git clone https://github.com/rhayes777/PyAutoFit
	cd packages && git clone https://github.com/Jammy2211/PyAutoArray
	cd packages && git clone https://github.com/Jammy2211/PyAutoGalaxy
	cd packages && git clone https://github.com/Jammy2211/PyAutoLens
.PHONY: packages


# remove temporary files
clean:
	rm -rf dataset/* *.log *.swp
	rm -rf latex/*.log latex/*.aux latex/*.bbl latex/*.blg latex/*.pdf
.PHONY: clean


# remove temporary files, the report, the model, and the figures
clean_all: clean
	rm -rf *.h5 *.pdf figures/*
.PHONY: clean_all


# remove the packages
clean_packages:
	rm -rf packages
.PHONY: clean_packages