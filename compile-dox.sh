#!/bin/bash

doxyfile="Doxyfile"
doxydocs="docs"
doxylatx="$doxydocs/latex"
outpdf="refman.pdf"

## brief commands executed
#	 doxygen $doxyfile
#  cd $doxylatx
#  make
## This will generate a file 'refman.pdf' which is the pdf manual

if [ ! -f "$doxyfile" ]; then
    echo "ERROR 1: $doxygen is not in the directory"
    exit
fi

doxygen "$doxyfile"

if [ ! -d "$doxydocs" ]; then
    echo "ERROR 2: $doxydocs is not in the directory"
    echo "Maybe the compilation directory has been changed in $doxyfile"
    exit
fi

if [ ! -d "$doxylatx" ]; then
    echo "ERROR 3: $doxylatx does not exist"
    echo "Maybe the latex directory has been changed in $doxyfile"
    exit
fi

current=`pwd`
cd "$doxylatx"
make
cp "$outpdf" "$current/$outpdf"
