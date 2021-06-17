default: report.pdf
.PHONY : default

report.pdf: latex/report.tex figures/loss_epochs.jpg
	cd latex
	pdflatex report.tex
	bibtex
	pdflatex report.tex
	bibtex
	pdflatex report.tex
	cd ..
	mv latex/report.pdf .


packages:
	mkdir packages
	cd packages
	git clone https://github.com/rhayes777/PyAutoFit
	git clone https://github.com/Jammy2211/PyAutoArray
	git clone https://github.com/Jammy2211/PyAutoGalaxy
	git clone https://github.com/Jammy2211/PyAutoLens
	cd ..


dataset/param_df.csv: packages
	#./rescal ...
	cp rescal-snow/scripts/SNO00018_t0.csp.gz .


clean:

.PHONY : clean

clean_all:

.PHONY : clean_all
