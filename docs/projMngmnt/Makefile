# Makefile
# From https://danielkaes.wordpress.com/2009/03/14/compiling-latex-documents-using-makefiles/

TEX=pdflatex
BIBTEX=bibtex

TEXFILES=$(wildcard *.tex)
PDFS=$(TEXFILES:.tex=.pdf)

all: $(PDFS)

%.pdf: %.tex
	$(TEX) $<
#	$(BIBTEX) $*
#	$(TEX) $<
	$(TEX) $<

clean-all:
	rm -f *.dvi *.log *.bak *.aux *.bbl *.blg *.idx *.ps *.eps *.pdf *.toc *.out *~

clean:
	rm -f *.log *.bak *.aux *.bbl *.blg *.idx *.toc *.out *~