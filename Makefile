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


# remove temporary files
clean:
	rm -rf dataset/* *.log *.swp
	rm -rf latex/*.log latex/*.aux latex/*.bbl latex/*.blg latex/*.pdf
.PHONY: clean


# remove temporary files, the report, the model, and the figures
reset: clean
	rm -rf *.h5 *.pdf figures dataset $(wildcard __pycache__)
.PHONY: reset
