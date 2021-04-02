#!/bin/bash
if [ -z $1 ]
then
    echo "usage: ./build_report.sh <lab_dir>"
    exit 1
fi

dir=$1
if [ -d $dir ]
then
    cd $dir
    if [ -f "report.ipynb" ]
    then
        jupyter nbconvert --to=latex --LatexExporter.template_file=../russian_template.tex.j2 --TemplateExporter.exclude_input=True report.ipynb
        pdflatex report.tex > /dev/null
        rm report.aux report.log report.out report.tex
        rm -rf report_files
    else
        echo "$dir/report.ipynb file not found"
        exit 1
    fi
else
    echo "$dir directory not found"
    exit 1
fi