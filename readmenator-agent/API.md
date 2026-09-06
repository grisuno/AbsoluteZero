# API

## app.py

### __post_init__ `def __post_init__(self)`
- Defined: `app.py:75`

### _compute_hash `def _compute_hash(self)`
- Defined: `app.py:81`
- Doc: Computa hash único para detectar duplicados.

### success_rate `def success_rate(self)`
- Defined: `app.py:87`

### to_dict `def to_dict(self)`
- Defined: `app.py:90`

### __init__ `def __init__(self, persist_path)`
- Defined: `app.py:96`

### add_task `def add_task(self, task)`
- Defined: `app.py:103`
- Doc: Añade tarea si no es duplicada y cumple criterios de calidad.

### _is_too_similar `def _is_too_similar(self, new_task)`
- Defined: `app.py:120`
- Doc: Verifica si la tarea es muy similar a las existentes.

### _simple_embedding `def _simple_embedding(self, program)`
- Defined: `app.py:137`
- Doc: Crea embedding simple del programa.

### _maintain_buffer_size `def _maintain_buffer_size(self, task_type)`
- Defined: `app.py:149`
- Doc: Mantiene el tamaño del buffer removiendo tareas con peor rendimiento.

### get_reference_tasks `def get_reference_tasks(self, task_type, n)`
- Defined: `app.py:156`
- Doc: Obtiene tareas de referencia con muestreo inteligente.

### save_to_disk `def save_to_disk(self)`
- Defined: `app.py:167`
- Doc: Guarda el banco de memoria en disco.

### load_from_disk `def load_from_disk(self)`
- Defined: `app.py:175`
- Doc: Carga el banco de memoria desde disco.

### __init__ `def __init__(self)`
- Defined: `app.py:188`

### update_performance `def update_performance(self, reward)`
- Defined: `app.py:198`
- Doc: Actualiza rendimiento y ajusta dificultad.

### _adjust_difficulty `def _adjust_difficulty(self, avg_performance)`
- Defined: `app.py:205`
- Doc: Ajusta dificultad basada en rendimiento.

### __init__ `def __init__(self)`
- Defined: `app.py:220`

### _initialize_with_seed `def _initialize_with_seed(self)`
- Defined: `app.py:237`
- Doc: Inicializa con triplete semilla mejorado.

### do_analyze `def do_analyze(self, args)`
- Defined: `app.py:258`
- Doc: Analiza código con opciones avanzadas.

### do_stats `def do_stats(self, arg)`
- Defined: `app.py:267`
- Doc: Muestra estadísticas del sistema.

### do_curriculum `def do_curriculum(self, arg)`
- Defined: `app.py:282`
- Doc: Muestra estado del curriculum learning.

### do_save_memory `def do_save_memory(self, arg)`
- Defined: `app.py:289`
- Doc: Guarda manualmente el banco de memoria.

### _analyze_sequential `def _analyze_sequential(self, directory, iterations)`
- Defined: `app.py:296`
- Doc: Análisis secuencial mejorado.

### _analyze_parallel `def _analyze_parallel(self, directory, iterations)`
- Defined: `app.py:309`
- Doc: Análisis paralelo con threading.

### _get_code_files `def _get_code_files(self, directory)`
- Defined: `app.py:326`
- Doc: Obtiene lista de archivos de código.

### _analyze_code_file `def _analyze_code_file(self, file_path, iterations)`
- Defined: `app.py:335`
- Doc: Análisis mejorado de archivo de código.

### _absolute_zero_loop `def _absolute_zero_loop(self, code_content, file_path, iterations)`
- Defined: `app.py:346`
- Doc: Ciclo principal mejorado con métricas.

### _propose_tasks_adaptive `def _propose_tasks_adaptive(self, code_content, file_path)`
- Defined: `app.py:364`
- Doc: Propone tareas con dificultad adaptativa.

### _build_adaptive_propose_prompt `def _build_adaptive_propose_prompt(self, task_type, code_content, ref_tasks, difficulty)`
- Defined: `app.py:393`
- Doc: Construye prompt adaptativo basado en dificultad.

### _query_deepseek_improved `def _query_deepseek_improved(self, prompt)`
- Defined: `app.py:423`
- Doc: Consulta API con manejo de errores mejorado y reintentos.

### _validate_task_improved `def _validate_task_improved(self, task)`
- Defined: `app.py:457`
- Doc: Validación mejorada con más verificaciones.

### _is_executable `def _is_executable(self, program)`
- Defined: `app.py:482`
- Doc: Verifica si el programa es ejecutable.

### _validate_deduction_task `def _validate_deduction_task(self, task)`
- Defined: `app.py:490`
- Doc: Valida tarea de deducción.

### _validate_abduction_task `def _validate_abduction_task(self, task)`
- Defined: `app.py:504`
- Doc: Valida tarea de abducción.

### _validate_induction_task `def _validate_induction_task(self, task)`
- Defined: `app.py:516`
- Doc: Valida tarea de inducción.

### _extract_function_name `def _extract_function_name(self, program)`
- Defined: `app.py:533`
- Doc: Extrae nombre de función del programa.

### _solve_tasks_improved `def _solve_tasks_improved(self, tasks, file_path)`
- Defined: `app.py:538`
- Doc: Resolución mejorada de tareas con métricas detalladas.

### _build_solve_prompt_improved `def _build_solve_prompt_improved(self, task)`
- Defined: `app.py:577`
- Doc: Construye prompt mejorado de resolución.

### _extract_solution_improved `def _extract_solution_improved(self, response, task_type)`
- Defined: `app.py:617`
- Doc: Extrae solución con múltiples estrategias.

### _calculate_reward_improved `def _calculate_reward_improved(self, task, solution)`
- Defined: `app.py:634`
- Doc: Cálculo mejorado de recompensa con múltiples factores.

### _verify_deduction_solution `def _verify_deduction_solution(self, task, solution)`
- Defined: `app.py:657`
- Doc: Verifica solución de deducción.

### _verify_abduction_solution `def _verify_abduction_solution(self, task, solution)`
- Defined: `app.py:669`
- Doc: Verifica solución de abducción.

### _verify_induction_solution `def _verify_induction_solution(self, task, solution)`
- Defined: `app.py:681`
- Doc: Verifica solución de inducción.

### _calculate_average_reward `def _calculate_average_reward(self, solutions)`
- Defined: `app.py:704`
- Doc: Calcula recompensa promedio de todas las soluciones.

### _update_model_improved `def _update_model_improved(self, tasks, solutions)`
- Defined: `app.py:712`
- Doc: Actualización mejorada del modelo con normalización adaptativa.

### __init__ `def __init__(self, memory_bank, cmd_instance)`
- Defined: `app.py:747`

### run_tournament `def run_tournament(self, rounds)`
- Defined: `app.py:753`
- Doc: Ejecuta torneo entre diferentes versiones del modelo.

### _select_tournament_tasks `def _select_tournament_tasks(self, n_tasks)`
- Defined: `app.py:778`
- Doc: Selecciona tareas diversas para el torneo.

### _evaluate_tournament_round `def _evaluate_tournament_round(self, tasks)`
- Defined: `app.py:797`
- Doc: Evalúa una ronda del torneo.

### _evaluate_single_task `def _evaluate_single_task(self, task)`
- Defined: `app.py:821`
- Doc: Evalúa una tarea individual con el modelo actual.

### _evaluate_baseline_task `def _evaluate_baseline_task(self, task)`
- Defined: `app.py:831`
- Doc: Evalúa con un modelo baseline simple.

### _update_elo_ratings `def _update_elo_ratings(self, results)`
- Defined: `app.py:841`
- Doc: Actualiza ratings ELO basado en resultados.

### _display_tournament_summary `def _display_tournament_summary(self)`
- Defined: `app.py:857`
- Doc: Muestra resumen del torneo.

### __init__ `def __init__(self)`
- Defined: `app.py:874`

### optimize_hyperparameters `def optimize_hyperparameters(self, cmd_instance, iterations)`
- Defined: `app.py:887`
- Doc: Optimiza hiperparámetros usando búsqueda aleatoria.

### _generate_hyperparameters `def _generate_hyperparameters(self)`
- Defined: `app.py:917`
- Doc: Genera nuevos hiperparámetros usando búsqueda aleatoria.

### _apply_hyperparameters `def _apply_hyperparameters(self, cmd_instance, hyperparams)`
- Defined: `app.py:927`
- Doc: Aplica hiperparámetros al sistema.

### _evaluate_performance `def _evaluate_performance(self, cmd_instance)`
- Defined: `app.py:936`
- Doc: Evalúa performance del sistema con hiperparámetros actuales.

### _generate_test_tasks `def _generate_test_tasks(self)`
- Defined: `app.py:952`
- Doc: Genera tareas de prueba para evaluación.

### _display_optimization_summary `def _display_optimization_summary(self)`
- Defined: `app.py:976`
- Doc: Muestra resumen de optimización.

### __init__ `def __init__(self)`
- Defined: `app.py:987`

### track_task_creation `def track_task_creation(self, task)`
- Defined: `app.py:996`
- Doc: Rastrea creación de tareas.

### track_performance `def track_performance(self, reward, task_type)`
- Defined: `app.py:1005`
- Doc: Rastrea performance del sistema.

### track_learning_velocity `def track_learning_velocity(self, success_rate)`
- Defined: `app.py:1013`
- Doc: Rastrea velocidad de aprendizaje.

### track_error `def track_error(self, error_type, context)`
- Defined: `app.py:1020`
- Doc: Rastrea errores del sistema.

### generate_report `def generate_report(self)`
- Defined: `app.py:1027`
- Doc: Genera reporte de analytics detallado.

### save_analytics `def save_analytics(self, filepath)`
- Defined: `app.py:1056`
- Doc: Guarda reporte de analytics.

### __init__ `def __init__(self)`
- Defined: `app.py:1067`

### do_tournament `def do_tournament(self, arg)`
- Defined: `app.py:1076`
- Doc: Ejecuta torneo de auto-juego.

### do_optimize `def do_optimize(self, arg)`
- Defined: `app.py:1084`
- Doc: Optimiza hiperparámetros del sistema.

### do_analytics `def do_analytics(self, arg)`
- Defined: `app.py:1092`
- Doc: Genera y muestra reporte de analytics.

### do_export_tasks `def do_export_tasks(self, arg)`
- Defined: `app.py:1100`
- Doc: Exporta tareas a archivo JSON.

### do_import_tasks `def do_import_tasks(self, arg)`
- Defined: `app.py:1111`
- Doc: Importa tareas desde archivo JSON.

### do_benchmark `def do_benchmark(self, arg)`
- Defined: `app.py:1132`
- Doc: Ejecuta benchmark de performance del sistema.

### _create_benchmark_tasks `def _create_benchmark_tasks(self)`
- Defined: `app.py:1171`
- Doc: Crea conjunto estándar de tareas para benchmark.

### _absolute_zero_loop `def _absolute_zero_loop(self, code_content, file_path, iterations)`
- Defined: `app.py:1199`
- Doc: Ciclo principal con analytics integrados.
