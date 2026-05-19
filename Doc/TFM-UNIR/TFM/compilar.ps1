Write-Host "Eliminando archivos temporales de compilación..."
Remove-Item -Path *.aux, *.log, *.toc, *.out -ErrorAction SilentlyContinue

Write-Host "Compilando el documento LaTeX..."
pdflatex TFM_IA_72858458R.tex
bibtex TFM_IA_72858458R.aux
pdflatex TFM_IA_72858458R.tex
pdflatex TFM_IA_72858458R.tex
Write-Host "Compilación completada."