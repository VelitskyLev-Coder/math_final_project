TEX_DIR := tex
TEX_MAIN := main.tex
PDFLATEX := pdflatex

.PHONY: all pdf clean

all: pdf

pdf:
	cd $(TEX_DIR) && $(PDFLATEX) -interaction=nonstopmode -halt-on-error $(TEX_MAIN)

clean:
	cd $(TEX_DIR) && powershell -NoProfile -Command "Remove-Item -Force -ErrorAction SilentlyContinue *.aux,*.log,*.out,*.toc,*.fls,*.fdb_latexmk,*.synctex.gz"
