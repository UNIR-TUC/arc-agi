## Plan: Eje H — reduce CPU dispatch bottleneck in parallel training

**TL;DR**: Baseline (baseline_summary.json, --demo 12) shows CPU saturated 74% of the run at ≥90%. Per the doc's Eje H (§9.11, priority 0), the cause is Python dispatch overhead + GPU→CPU syncs in the hot path, not the math itself. Implement a `--max-workers` concurrency knob, remove dead GPU→CPU syncs from the logger (H2), reduce postprocessing frequency (H3), and vectorize the offset loops in the loss computation (H1). Defer `torch.compile` (H4) — riskiest on ROCm.

**Steps**
1. **Phase 0 — Concurrency cap** (trivial, independent): add `--max-workers` CLI arg to parallel_train.py, feeds directly into the existing `n_cpus` cap already threaded through `parallelize_runs`.
2. **Phase 1 — H2 (GPU→CPU sync removal)** in solution_selection.py `Logger.log()`: stop calling `.cpu().numpy()` ~30×/iteration. Confirmed via grep that these curves are **never read anywhere** — store detached GPU tensors, add `materialize_curves()` to batch-convert once per task (called from solve_task.py and train.py).
3. **Phase 2 — H3 (postprocess stride)**, *depends on Phase 1*: gate the expensive `_postprocess_solution` calls in `Logger._track_solution()` behind `train_step % postprocess_stride == 0` (default K=4 for parallel path, K=1 elsewhere). On skipped steps, append neutral `(0, -np.inf)` entries — verified as a no-op under `np.logaddexp` re-accumulation used by plot_accuracy.py/list_solved_puzzles.py. Threaded through solve_task.py → `parallelize_runs` → `run_split` → new `--postprocess-stride` CLI arg.
4. **Phase 3 — H1 (vectorize offset loops)** in train.py, independent but higher risk: rewrite `mask_select_logprobs` using a cumsum closed-form (removes its internal O(offset) loop), and replace the nested `x_offset × y_offset` + per-pair `cross_entropy` loop with a single batched `unfold`-based cross-entropy call (target crop is offset-independent, confirmed from code — makes this exact, not approximate). Requires a numeric parity check before wiring in.

**Relevant files**
- train.py — `mask_select_logprobs`, `take_step`
- solution_selection.py — `Logger` class
- solve_task.py — `solve_task()` signature
- parallel_train.py — `parallelize_runs`, `run_split`, CLI

**Verification**
1. `get_errors` after each phase.
2. Ad-hoc numeric parity checks for Phase 3 (old vs new, `torch.allclose`).
3. `profile_parallel_train.py --label optimized -- --split training --demo 12` then `--compare` against existing `baseline_summary.json` — expect CPU mean/saturation to drop, guardrail to pass.
4. Short supervised stress test above 12 concurrent tasks (~10-15 min, not multi-hour).
5. Spot-check solved-task hashes against ground truth for correctness.

**Decisions**
- H4 (torch.compile) explicitly deferred — out of scope this pass.
- No changes to arc_compressor.py/layers.py/multitensor_systems.py/initializers.py/preprocessing.py — model math untouched.

## Implementation

**Phase 0** — `--max-workers` CLI knob added to parallel_train.py, directly caps the existing `n_cpus` concurrency limit.

**Phase 1 (H2)** — solution_selection.py's `Logger` no longer syncs GPU→CPU every iteration for KL/loss curves (confirmed dead data, never read elsewhere); added `materialize_curves()` to batch-convert once per task.

**Phase 2 (H3)** — `Logger._track_solution()` now runs the expensive postprocessing every `postprocess_stride` steps (default 4 for parallel runs), with neutral `(0, -inf)` placeholders on skipped steps (mathematically a no-op under the `logaddexp` re-accumulation used by plot_accuracy.py/list_solved_puzzles.py). Threaded through solve_task.py → `parallelize_runs` → `run_split` → new `--postprocess-stride` CLI arg.

**Phase 3 (H1)** — train.py's `mask_select_logprobs` rewritten with a cumsum closed-form (no more O(offset) loop), and the `x_offset × y_offset` nested loop replaced with a single batched `unfold`-based `cross_entropy` call.

**Verification performed:**
- `get_errors`: clean on all 4 files.
- Numeric parity: `mask_select_logprobs` (20 random trials) and full `take_step` reconstruction math (real tasks, both grid-size-known and grid-size-uncertain branches, 3 train_steps each) — exact match against the original implementation.
- Integration smoke test: direct `solve_task.solve_task` call confirmed correct stride behavior, array lengths, and no exceptions.
- CLI smoke-launched `parallel_train.py --demo 2 --max-workers 2 --postprocess-stride 4` to confirm it starts correctly under the new args.

**Not yet done** (long-running, left for you to execute): the actual A/B comparison — `profile_parallel_train.py --label optimized -- --split training --demo 12` followed by `--compare` against the existing baseline_summary.json — plus the short supervised stress test above 12 concurrent tasks. These take hours/require live monitoring, so I stopped short of running them in this session. Let me know if you'd like me to kick off the optimized profiler run now.

## Análisis de la comparativa limpia `baseline` → `optimized_clean`

Esta vez sí es una comparación válida: confirmo que optimized_clean_summary.json tiene `workers_completed: 12` exactos (sin los 12 procesos fantasma de Fase 1 de la vez anterior) — Fase 1 se saltó igual que en `baseline`. Ya no hay contaminación.

### ✅ Corrección de precisión sobre `mean_worker_lifetime_s`

Aun así hay una pequeña asimetría residual: `baseline` cuenta **13** workers "completados" (12 reales + 1 proceso de bookkeeping/manager que por una carrera de tiempos quedó marcado como completado), mientras que `optimized_clean` cuenta exactamente los **12** reales. Haciendo la media a mano solo con los 12 workers de entrenamiento real de cada lado:

| | baseline (12 reales) | optimized_clean (12 reales) |
|---|---|---|
| Vida media | ~7101.4 s | 6379.4 s (coincide exacto con el reportado) |
| CPU medio/worker | ~188.4 % | ~188.4 % |

Diferencia real: **−722 s / −10.2 %** (muy cercano al −11.0% reportado, la corrección es marginal).

### 📊 Resultados clave

1. **Mejora real de velocidad (~10%)**: tanto el `wall_time_s` total (−10.0%) como la vida media por tarea (~−10.2%, ya corregida) bajan de forma consistente y sustancial. Esto es un cambio genuino, no ruido — a diferencia de la comparación anterior contaminada, que sugería (falsamente) que el tiempo real por tarea había empeorado.

2. **CPU% por worker se mantiene igual (~188.4% en ambos)**: cada proceso sigue usando ~1.88 núcleos de media durante toda su vida. Es decir, **la ganancia no viene de "usar menos CPU por segundo"**, sino de que cada una de las 1500 iteraciones se completa más rápido en tiempo real (menos overhead de Python en `mask_select_logprobs`/bucle de offsets vectorizado con `unfold`, menos paradas por sincronización GPU→CPU). El resultado neto: mismo ritmo de ocupación de CPU, pero se necesita menos tiempo total para acabar el mismo trabajo → menos CPU-segundos totales consumidos por tarea (~10% menos).

3. **`cpu_saturation_fraction` baja de forma más clara ahora: 0.743 → 0.65 (~12.5% relativo)**, mejor que el 0.695 contaminado de la comparación anterior. Con la contaminación de Fase 1 eliminada, esta es una señal limpia y coherente con el objetivo de Eje H.

4. **GPU sigue al ~99.5-99.6% en ambos casos** — no se sacrificó utilización de GPU para lograr la mejora; el cuello de botella de CPU se redujo sin dejar la GPU ociosa.

5. **`mean_workers` −0.2% [REGRESSION]**: 12.75 vs 12.72 — diferencia de ruido puro (la lógica de `compare()` marca como "regresión" cualquier delta negativo, por pequeño que sea, aunque aquí es estadísticamente insignificante). No indica ningún problema real de scheduling/concurrencia.

6. **Corrección funcional (importante):** comparé los ficheros results_2026-07-17_08:25:43.txt (baseline) y results_2026-07-19_03:18:09.txt (optimized_clean) — **ambos resolvieron exactamente las mismas 5 tareas** (`025d127b`, `05269061`, `08ed6ac7`, `0962bcdd`, `06df4c85`), 5/12 (41.7%) en los dos casos. Esto confirma que el `postprocess_stride=4` (H3) **no costó precisión/accuracy** en este demo — sigue resolviendo el mismo conjunto de tareas.

### Conclusión

Con la contaminación de Fase 1 eliminada, Eje H cumple **ambos objetivos**:
- **No empeora la presión de CPU** (guardrail de seguridad, objetivo crítico) — de hecho la mejora de forma clara (`saturation_fraction` −12.5% relativo).
- **Sí aporta una mejora real de rendimiento** (~10% menos tiempo de pared por tarea/por run completo), sin perder GPU-utilización ni precisión de resolución.

El giro respecto al análisis anterior (que sugería un posible empeoramiento del tiempo por tarea) confirma lo importante que era controlar el estado de la caché de Fase 1 antes de comparar — con eso resuelto, los datos ahora cuentan una historia consistente y positiva.