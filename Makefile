all:
	rm -f _book/*.html &&\
	BOOKDOWN_FULL_PDF=false Rscript --quiet _render.R

pdf:
	Rscript _render.R "bookdown::pdf_book"

gitbook:
	Rscript _render.R "bookdown::gitbook"

pdf2:
	BOOKDOWN_FULL_PDF=false Rscript --quiet _render.R "bookdown::pdf_book"
