.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib neurips_2025.sty
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out main.synctex.gz
