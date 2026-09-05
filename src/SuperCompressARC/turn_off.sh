#!/bin/bash

# Esperar 3 horas
sleep 3h

# Obtener uso de CPU durante 10 segundos y calcular la media
CPU_USAGE=$(mpstat 10 1 | awk '/Average:/ {print 100 - $NF}')

echo "Uso medio de CPU: ${CPU_USAGE}%"

# Apagar si está por debajo del 20%
if awk "BEGIN {exit !($CPU_USAGE < 20)}"; then
    echo "CPU por debajo del 20%. Apagando..."
    sudo shutdown -h now
else
    echo "CPU por encima o igual al 20%. No se hace nada."
fi