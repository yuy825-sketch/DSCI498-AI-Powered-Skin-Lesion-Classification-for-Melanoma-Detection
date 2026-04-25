TEX=main.tex
OUTDIR=build

.PHONY: all pdf clean watch

all: pdf

pdf:
	@mkdir -p $(OUTDIR)
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -outdir=$(OUTDIR) $(TEX); \
	else \
		echo "latexmk not found; falling back to pdflatex/bibtex."; \
		pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=$(OUTDIR) $(TEX); \
		if grep -q '\\\\citation' $(OUTDIR)/main.aux 2>/dev/null; then (cd $(OUTDIR) && bibtex main); fi; \
		pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=$(OUTDIR) $(TEX); \
		pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=$(OUTDIR) $(TEX); \
	fi

watch:
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -pvc -outdir=$(OUTDIR) $(TEX); \
	else \
		echo "latexmk not found; watch mode unavailable."; \
		exit 2; \
	fi

clean:
	latexmk -C -outdir=$(OUTDIR) $(TEX) || true
	rm -rf $(OUTDIR)
