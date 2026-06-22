---
name: latex-espanol
description: 'Escribe y revisa un TFM/tesis UNIR en espanol con LaTeX. Use when drafting secciones academicas, bibliografia, citas APA con apacite, resumenes, abstracts, compilacion y correccion de documentos .tex en este repositorio.'
argument-hint: 'Seccion, objetivo o tarea de redaccion en LaTeX'
---

# Escritura En Espanol Con LaTeX

## Que Produce
- Texto academico en espanol listo para integrar en archivos `.tex`.
- Ajustes de estructura, citas y formato coherentes con el flujo TFM/UNIR de este repositorio.
- Comandos concretos para compilar y verificar el documento cuando haga falta.

## Cuando Usar
- Redactar o reescribir capitulos, secciones o subsecciones en LaTeX.
- Preparar resumen, abstract, introduccion, objetivos, estado del arte o conclusiones.
- Insertar citas narrativas, parenteticas, literales cortas o bloques de cita.
- Revisar espanol academico, cohesión, tono formal y consistencia terminologica.
- Corregir problemas comunes de compilacion, bibliografia o referencias cruzadas.

## Alcance
- Esta skill es especifica para el TFM/UNIR de este repositorio.
- Prioriza `apacite`, la estructura del archivo actual y la compilacion local existente.
- No debe reemplazar el flujo por paquetes, estilos o comandos alternativos salvo peticion explicita.

## Contexto Del Repositorio
- El trabajo principal esta en `Doc/TFM-UNIR/TFM/TFM_IA_72858458R.tex`.
- La compilacion habitual se hace con `Doc/TFM-UNIR/TFM/compilar.sh`.
- El proyecto usa `apacite` y dispone de ejemplos en `Doc/TFM-UNIR/TFM/manual_citas.md`.
- El documento usa clase `book`, idioma `spanish` y convenciones de TFM/UNIR.

## Procedimiento
1. Identifica la salida exacta que se necesita.
   Pide o infiere si el usuario quiere redaccion nueva, reescritura, resumen, traduccion, insercion de citas o correccion de un fragmento existente.
2. Ubica la seccion y conserva la convencion local.
   Mantén comandos, estructura de capitulos/secciones, tono academico y estilo del documento existente. No cambies paquetes, clase, macros ni flujo bibliografico salvo que el usuario lo pida.
3. Redacta primero el contenido y despues la sintaxis LaTeX.
   Escribe en espanol claro, formal y preciso. Luego encapsula el texto con la estructura LaTeX minima necesaria: `\chapter`, `\section`, `\subsection`, `\begin{itemize}`, `\cite{...}`, `\citeA{...}`, `\begin{quote}`.
4. Elige el patron de cita correcto.
   Usa `\cite{clave}` para cita parentetica, `\citeA{clave}` cuando el autor forme parte de la frase, `\cite[p.~xx]{clave}` para pagina concreta y bloque `quote` para citas largas.
5. Ajusta el contenido al tipo de seccion.
   Para `Resumen` y `Abstract`, prioriza objetivo, metodologia, resultados y conclusiones. Para introduccion y estado del arte, prioriza contexto, problema, justificacion y relacion entre trabajos.
6. Revisa calidad linguistica y academica.
   Elimina repeticiones, ambiguedades, cambios bruscos de registro, afirmaciones infladas y traducciones literales torpes. Prefiere frases directas y conectores precisos.
7. Verifica integracion tecnica.
   Comprueba que llaves, entornos, comandos y claves BibTeX sean plausibles y consistentes con `apacite`.
8. Si hubo cambios en archivos LaTeX, compila para validar.
   Desde `Doc/TFM-UNIR/TFM/`, ejecuta `./compilar.sh` o la secuencia `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`.

## Decisiones Y Ramas

### Si el usuario pide redaccion nueva
- Solicita o deduce tema, objetivo de la seccion, longitud aproximada y referencias obligatorias.
- Entrega texto listo para pegar en LaTeX, no solo ideas sueltas.

### Si el usuario pide reescritura
- Conserva el significado tecnico.
- Mejora claridad, cohesion, estilo y correccion del espanol sin introducir afirmaciones no respaldadas.

### Si el usuario pide citas
- Usa el comando de cita minimo que satisfaga el caso.
- Si falta la clave BibTeX, deja un marcador claro y dilo explicitamente.

### Si el usuario pide resumen o abstract
- Mantén la extension breve.
- Incluye objetivo, enfoque, resultado y cierre.
- Si el documento exige dos idiomas, no mezcles espanol e ingles dentro del mismo bloque.

### Si hay errores de compilacion
- Prioriza primero errores sintacticos de LaTeX.
- Despues revisa claves BibTeX, referencias cruzadas y paquetes.
- No cambies el flujo de compilacion si el problema es local a una cita o entorno.

## Criterios De Calidad
- Espanol academico natural, sin calcos innecesarios del ingles.
- Terminologia coherente en todo el fragmento.
- Sintaxis LaTeX minima y correcta.
- Citas congruentes con `apacite`.
- Parrafos con una idea dominante y transiciones claras.
- Ningun marcador pendiente sin explicar.
- Todas las palabras que sean en inglés deben ir en cursiva salvo nombres propios, siglas o comandos LaTeX.

## Comprobacion Final
- El texto responde exactamente a la seccion pedida.
- El tono es formal y consistente con un TFM.
- Los comandos LaTeX usados son compatibles con el archivo actual.
- Las citas tienen clave, formato y pagina cuando corresponde.
- Si se editaron archivos, el documento compila o quedan descritos los errores residuales.

## Respuesta Esperada
- Devuelve primero el fragmento LaTeX final.
- Si hiciste supuestos, listalos despues en una nota breve.
- Si detectas una mejora estructural mayor, separala claramente del texto principal para que el usuario decida si aplicarla.

## Documentación Fundamental
- Si es necesario haz uso de la carpeta `Doc/Doc-markdown/` con todos los archivos de referencia y notas adicionales en formato Markdown.
- Consulta si es necesario todas las citas disponibles en `Doc/TFM-UNIR/TFM/bibliografia.bib` y el manual de estilo `Doc/TFM-UNIR/TFM/manual_citas.md`.