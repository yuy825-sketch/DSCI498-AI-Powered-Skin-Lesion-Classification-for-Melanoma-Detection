# Minimal latexmk config for IEEE-style projects
$pdf_mode = 1;
$out_dir = 'build';
$aux_dir = 'build';

$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error -file-line-error %O %S';
$bibtex   = 'bibtex %O %B';

$clean_ext = 'aux bbl blg fdb_latexmk fls log out toc synctex.gz';

