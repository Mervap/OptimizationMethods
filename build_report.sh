#!/bin/bash
cd "$1"
jupyter nbconvert --to=latex --LatexExporter.template_file=../russian_template.tex.j2 --TemplateExporter.exclude_input=True report.ipynb
pdflatex report.tex > /dev/null
rm report.aux report.log report.out report.tex
rm -rf report_files