# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM.
> No LLMs. No tokens. Pure static analysis. See more [here](https://github.com/grisuno/ReadMenator)

**Total Files Parsed:** 2 | **Total Symbols Extracted:** 83 | **Total Imports:** 19

## Structural Knowledge Map
```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray:5 5,color:#aaa;
    app_py["app.py (py)"]
    class app_py mod;
    app_py_Task["Task"]
    class app_py_Task cls;
    app_py --> app_py_Task
    app_py_MemoryBank["MemoryBank"]
    class app_py_MemoryBank cls;
    app_py --> app_py_MemoryBank
    app_py_CurriculumLearning["CurriculumLearning"]
    class app_py_CurriculumLearning cls;
    app_py --> app_py_CurriculumLearning
    app_py_AbsoluteZeroCmd["AbsoluteZeroCmd"]
    class app_py_AbsoluteZeroCmd cls;
    app_py --> app_py_AbsoluteZeroCmd
    app_py_SelfPlayTraining["SelfPlayTraining"]
    class app_py_SelfPlayTraining cls;
    app_py --> app_py_SelfPlayTraining
    install_sh["install.sh (sh)"]
    class install_sh mod;
    ext_os["os"]
    class ext_os ext;
    app_py -.->|imports| ext_os
    ext_json["json"]
    class ext_json ext;
    app_py -.->|imports| ext_json
    ext_random["random"]
    class ext_random ext;
    app_py -.->|imports| ext_random
    ext_re["re"]
    class ext_re ext;
    app_py -.->|imports| ext_re
    ext_logging["logging"]
    class ext_logging ext;
    app_py -.->|imports| ext_logging
    ext_requests["requests"]
    class ext_requests ext;
    app_py -.->|imports| ext_requests
    ext_collections["collections"]
    class ext_collections ext;
    app_py -.->|imports| ext_collections
    ext_cmd2["cmd2"]
    class ext_cmd2 ext;
    app_py -.->|imports| ext_cmd2
    ext_argparse["argparse"]
    class ext_argparse ext;
    app_py -.->|imports| ext_argparse
    ext_rich_console["rich.console"]
    class ext_rich_console ext;
    app_py -.->|imports| ext_rich_console
    ext_rich_progress["rich.progress"]
    class ext_rich_progress ext;
    app_py -.->|imports| ext_rich_progress
    ext_numpy["numpy"]
    class ext_numpy ext;
    app_py -.->|imports| ext_numpy
    ext_datetime["datetime"]
    class ext_datetime ext;
    app_py -.->|imports| ext_datetime
    ext_hashlib["hashlib"]
    class ext_hashlib ext;
    app_py -.->|imports| ext_hashlib
    ext_typing["typing"]
    class ext_typing ext;
    app_py -.->|imports| ext_typing
    ext_threading["threading"]
    class ext_threading ext;
    app_py -.->|imports| ext_threading
    ext_time["time"]
    class ext_time ext;
    app_py -.->|imports| ext_time
    ext_dataclasses["dataclasses"]
    class ext_dataclasses ext;
    app_py -.->|imports| ext_dataclasses
    ext_pickle["pickle"]
    class ext_pickle ext;
    app_py -.->|imports| ext_pickle
```

---

## Architecture Reference

### PY (1 files)

#### `app.py`
**Path:** `app.py`

**Classes:**
- `Task` (line 62) `class Task` - *Estructura mejorada para tareas con metadatos.*
- `MemoryBank` (line 93) `class MemoryBank` - *Sistema de memoria mejorado con persistencia y clustering.*
- `CurriculumLearning` (line 185) `class CurriculumLearning` - *Sistema de aprendizaje curricular que ajusta dificultad automáticamente.*
- `AbsoluteZeroCmd` (line 217) `class AbsoluteZeroCmd(Cmd)` - *Sistema Absolute Zero mejorado con características avanzadas.*
- `SelfPlayTraining` (line 744) `class SelfPlayTraining` - *Sistema de entrenamiento por auto-juego avanzado.*
- `MetaLearning` (line 871) `class MetaLearning` - *Sistema de meta-aprendizaje para optimización de hiperparámetros.*
- `AdvancedAnalytics` (line 984) `class AdvancedAnalytics` - *Sistema de análisis avanzado y visualización.*
- `AbsoluteZeroCmd` (line 1064) `class AbsoluteZeroCmd(AbsoluteZeroCmd)` - *Extensión de la clase principal con funcionalidades avanzadas.*

**Functions:**
- `__post_init__` (line 75) `def __post_init__(self)`
- `_compute_hash` (line 81) `def _compute_hash(self)` - *Computa hash único para detectar duplicados.*
- `success_rate` (line 87) `def success_rate(self)`
- `to_dict` (line 90) `def to_dict(self)`
- `__init__` (line 96) `def __init__(self, persist_path)`
- `add_task` (line 103) `def add_task(self, task)` - *Añade tarea si no es duplicada y cumple criterios de calidad.*
- `_is_too_similar` (line 120) `def _is_too_similar(self, new_task)` - *Verifica si la tarea es muy similar a las existentes.*
- `_simple_embedding` (line 137) `def _simple_embedding(self, program)` - *Crea embedding simple del programa.*
- `_maintain_buffer_size` (line 149) `def _maintain_buffer_size(self, task_type)` - *Mantiene el tamaño del buffer removiendo tareas con peor rendimiento.*
- `get_reference_tasks` (line 156) `def get_reference_tasks(self, task_type, n)` - *Obtiene tareas de referencia con muestreo inteligente.*
- `save_to_disk` (line 167) `def save_to_disk(self)` - *Guarda el banco de memoria en disco.*
- `load_from_disk` (line 175) `def load_from_disk(self)` - *Carga el banco de memoria desde disco.*
- `__init__` (line 188) `def __init__(self)`
- `update_performance` (line 198) `def update_performance(self, reward)` - *Actualiza rendimiento y ajusta dificultad.*
- `_adjust_difficulty` (line 205) `def _adjust_difficulty(self, avg_performance)` - *Ajusta dificultad basada en rendimiento.*
- `__init__` (line 220) `def __init__(self)`
- `_initialize_with_seed` (line 237) `def _initialize_with_seed(self)` - *Inicializa con triplete semilla mejorado.*
- `do_analyze` (line 258) `def do_analyze(self, args)` - *Analiza código con opciones avanzadas.*
- `do_stats` (line 267) `def do_stats(self, arg)` - *Muestra estadísticas del sistema.*
- `do_curriculum` (line 282) `def do_curriculum(self, arg)` - *Muestra estado del curriculum learning.*
- `do_save_memory` (line 289) `def do_save_memory(self, arg)` - *Guarda manualmente el banco de memoria.*
- `_analyze_sequential` (line 296) `def _analyze_sequential(self, directory, iterations)` - *Análisis secuencial mejorado.*
- `_analyze_parallel` (line 309) `def _analyze_parallel(self, directory, iterations)` - *Análisis paralelo con threading.*
- `_get_code_files` (line 326) `def _get_code_files(self, directory)` - *Obtiene lista de archivos de código.*
- `_analyze_code_file` (line 335) `def _analyze_code_file(self, file_path, iterations)` - *Análisis mejorado de archivo de código.*
- `_absolute_zero_loop` (line 346) `def _absolute_zero_loop(self, code_content, file_path, iterations)` - *Ciclo principal mejorado con métricas.*
- `_propose_tasks_adaptive` (line 364) `def _propose_tasks_adaptive(self, code_content, file_path)` - *Propone tareas con dificultad adaptativa.*
- `_build_adaptive_propose_prompt` (line 393) `def _build_adaptive_propose_prompt(self, task_type, code_content, ref_tasks, difficulty)` - *Construye prompt adaptativo basado en dificultad.*
- `_query_deepseek_improved` (line 423) `def _query_deepseek_improved(self, prompt)` - *Consulta API con manejo de errores mejorado y reintentos.*
- `_validate_task_improved` (line 457) `def _validate_task_improved(self, task)` - *Validación mejorada con más verificaciones.*
- `_is_executable` (line 482) `def _is_executable(self, program)` - *Verifica si el programa es ejecutable.*
- `_validate_deduction_task` (line 490) `def _validate_deduction_task(self, task)` - *Valida tarea de deducción.*
- `_validate_abduction_task` (line 504) `def _validate_abduction_task(self, task)` - *Valida tarea de abducción.*
- `_validate_induction_task` (line 516) `def _validate_induction_task(self, task)` - *Valida tarea de inducción.*
- `_extract_function_name` (line 533) `def _extract_function_name(self, program)` - *Extrae nombre de función del programa.*
- `_solve_tasks_improved` (line 538) `def _solve_tasks_improved(self, tasks, file_path)` - *Resolución mejorada de tareas con métricas detalladas.*
- `_build_solve_prompt_improved` (line 577) `def _build_solve_prompt_improved(self, task)` - *Construye prompt mejorado de resolución.*
- `_extract_solution_improved` (line 617) `def _extract_solution_improved(self, response, task_type)` - *Extrae solución con múltiples estrategias.*
- `_calculate_reward_improved` (line 634) `def _calculate_reward_improved(self, task, solution)` - *Cálculo mejorado de recompensa con múltiples factores.*
- `_verify_deduction_solution` (line 657) `def _verify_deduction_solution(self, task, solution)` - *Verifica solución de deducción.*
- `_verify_abduction_solution` (line 669) `def _verify_abduction_solution(self, task, solution)` - *Verifica solución de abducción.*
- `_verify_induction_solution` (line 681) `def _verify_induction_solution(self, task, solution)` - *Verifica solución de inducción.*
- `_calculate_average_reward` (line 704) `def _calculate_average_reward(self, solutions)` - *Calcula recompensa promedio de todas las soluciones.*
- `_update_model_improved` (line 712) `def _update_model_improved(self, tasks, solutions)` - *Actualización mejorada del modelo con normalización adaptativa.*
- `__init__` (line 747) `def __init__(self, memory_bank, cmd_instance)`
- `run_tournament` (line 753) `def run_tournament(self, rounds)` - *Ejecuta torneo entre diferentes versiones del modelo.*
- `_select_tournament_tasks` (line 778) `def _select_tournament_tasks(self, n_tasks)` - *Selecciona tareas diversas para el torneo.*
- `_evaluate_tournament_round` (line 797) `def _evaluate_tournament_round(self, tasks)` - *Evalúa una ronda del torneo.*
- `_evaluate_single_task` (line 821) `def _evaluate_single_task(self, task)` - *Evalúa una tarea individual con el modelo actual.*
- `_evaluate_baseline_task` (line 831) `def _evaluate_baseline_task(self, task)` - *Evalúa con un modelo baseline simple.*
- `_update_elo_ratings` (line 841) `def _update_elo_ratings(self, results)` - *Actualiza ratings ELO basado en resultados.*
- `_display_tournament_summary` (line 857) `def _display_tournament_summary(self)` - *Muestra resumen del torneo.*
- `__init__` (line 874) `def __init__(self)`
- `optimize_hyperparameters` (line 887) `def optimize_hyperparameters(self, cmd_instance, iterations)` - *Optimiza hiperparámetros usando búsqueda aleatoria.*
- `_generate_hyperparameters` (line 917) `def _generate_hyperparameters(self)` - *Genera nuevos hiperparámetros usando búsqueda aleatoria.*
- `_apply_hyperparameters` (line 927) `def _apply_hyperparameters(self, cmd_instance, hyperparams)` - *Aplica hiperparámetros al sistema.*
- `_evaluate_performance` (line 936) `def _evaluate_performance(self, cmd_instance)` - *Evalúa performance del sistema con hiperparámetros actuales.*
- `_generate_test_tasks` (line 952) `def _generate_test_tasks(self)` - *Genera tareas de prueba para evaluación.*
- `_display_optimization_summary` (line 976) `def _display_optimization_summary(self)` - *Muestra resumen de optimización.*
- `__init__` (line 987) `def __init__(self)`
- `track_task_creation` (line 996) `def track_task_creation(self, task)` - *Rastrea creación de tareas.*
- `track_performance` (line 1005) `def track_performance(self, reward, task_type)` - *Rastrea performance del sistema.*
- `track_learning_velocity` (line 1013) `def track_learning_velocity(self, success_rate)` - *Rastrea velocidad de aprendizaje.*
- `track_error` (line 1020) `def track_error(self, error_type, context)` - *Rastrea errores del sistema.*
- `generate_report` (line 1027) `def generate_report(self)` - *Genera reporte de analytics detallado.*
- `save_analytics` (line 1056) `def save_analytics(self, filepath)` - *Guarda reporte de analytics.*
- `__init__` (line 1067) `def __init__(self)`
- `do_tournament` (line 1076) `def do_tournament(self, arg)` - *Ejecuta torneo de auto-juego.*
- `do_optimize` (line 1084) `def do_optimize(self, arg)` - *Optimiza hiperparámetros del sistema.*
- `do_analytics` (line 1092) `def do_analytics(self, arg)` - *Genera y muestra reporte de analytics.*
- `do_export_tasks` (line 1100) `def do_export_tasks(self, arg)` - *Exporta tareas a archivo JSON.*
- `do_import_tasks` (line 1111) `def do_import_tasks(self, arg)` - *Importa tareas desde archivo JSON.*
- `do_benchmark` (line 1132) `def do_benchmark(self, arg)` - *Ejecuta benchmark de performance del sistema.*
- `_create_benchmark_tasks` (line 1171) `def _create_benchmark_tasks(self)` - *Crea conjunto estándar de tareas para benchmark.*
- `_absolute_zero_loop` (line 1199) `def _absolute_zero_loop(self, code_content, file_path, iterations)` - *Ciclo principal con analytics integrados.*

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*
