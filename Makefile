default: report.pdf
.PHONY : default


report.pdf: latex/report.tex figures/loss_epochs.jpg
	cd latex && pdflatex report.tex && bibtex report.aux && pdflatex report.tex && bibtex report.aux && pdflatex report.tex
	mv latex/report.pdf .


dataset/param_df.csv: 
	python generate_data.py --force


figures/loss_epochs.jpg: model_and_figures.py params_and_cli.py dataset/param_df.csv
	python model_and_figures.py


packages:
	mkdir packages
	cd packages && git clone https://github.com/rhayes777/PyAutoFit
	cd packages && git clone https://github.com/Jammy2211/PyAutoArray
	cd packages && git clone https://github.com/Jammy2211/PyAutoGalaxy
	cd packages && git clone https://github.com/Jammy2211/PyAutoLens


clean:
	rm -rf dataset/* *.log *.swp
	rm -rf latex/*.log latex/*.aux latex/*.bbl latex/*.blg latex/*.pdf
.PHONY: clean


clean_all:
	rm -rf dataset/* figures/* *.log *.swp
	rm -rf latex/*.log latex/*.aux latex/*.bbl latex/*.blg latex/*.pdf
	rm -rf *.h5 *.pdf
.PHONY: clean_all
