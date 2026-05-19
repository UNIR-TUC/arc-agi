#!/bin/bash
# Eliminar archivos temporales de compilación
echo "Eliminando archivos temporales de compilación..."
rm -f *.aux *.log *.toc *.out

echo "Compilando ..."

pdflatex TFM_IA_72858458R.tex
bibtex TFM_IA_72858458R.aux
pdflatex TFM_IA_72858458R.tex
pdflatex TFM_IA_72858458R.tex
echo "✔ Compilacion finalizada"