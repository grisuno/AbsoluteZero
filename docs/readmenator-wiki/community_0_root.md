# root

*Community 0 | 2 files | cohesion 1.00*

## Definition

This community groups 2 file(s) rooted at `root` with dominant language py (cohesion 1.00). Central symbols: `AbsoluteZeroCmd`, `AdvancedAnalytics`, `CurriculumLearning`, `MemoryBank`, `MetaLearning`, `SelfPlayTraining`, `Task`, `__init__`. Core file: `app.py` (83 symbols). Documented purpose: Autor: Gris Iscomeback Correo electrónico: grisiscomeback[at]gmail[dot]com Fecha de creación: xx/xx/xxxx Licencia: GPL v3  Descripción:.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `app.py` | py | utility | 83 | yes |
| `install.sh` | sh | utility | 0 | no |

## Key Symbols

- `Task` (class, `app.py:62`) `class Task` - Estructura mejorada para tareas con metadatos.
- `__post_init__` (method, `app.py:75`) `def __post_init__(self)`
- `_compute_hash` (method, `app.py:81`) `def _compute_hash(self)` - Computa hash único para detectar duplicados.
- `success_rate` (method, `app.py:87`) `def success_rate(self)`
- `to_dict` (method, `app.py:90`) `def to_dict(self)`
- `MemoryBank` (class, `app.py:93`) `class MemoryBank` - Sistema de memoria mejorado con persistencia y clustering.
- `__init__` (method, `app.py:96`) `def __init__(self, persist_path)`
- `add_task` (method, `app.py:103`) `def add_task(self, task)` - Añade tarea si no es duplicada y cumple criterios de calidad.
- `_is_too_similar` (method, `app.py:120`) `def _is_too_similar(self, new_task)` - Verifica si la tarea es muy similar a las existentes.
- `_simple_embedding` (method, `app.py:137`) `def _simple_embedding(self, program)` - Crea embedding simple del programa.
- `_maintain_buffer_size` (method, `app.py:149`) `def _maintain_buffer_size(self, task_type)` - Mantiene el tamaño del buffer removiendo tareas con peor rendimiento.
- `get_reference_tasks` (method, `app.py:156`) `def get_reference_tasks(self, task_type, n)` - Obtiene tareas de referencia con muestreo inteligente.
- `save_to_disk` (method, `app.py:167`) `def save_to_disk(self)` - Guarda el banco de memoria en disco.
- `load_from_disk` (method, `app.py:175`) `def load_from_disk(self)` - Carga el banco de memoria desde disco.
- `CurriculumLearning` (class, `app.py:185`) `class CurriculumLearning` - Sistema de aprendizaje curricular que ajusta dificultad automáticamente.
- `__init__` (method, `app.py:188`) `def __init__(self)`
- `update_performance` (method, `app.py:198`) `def update_performance(self, reward)` - Actualiza rendimiento y ajusta dificultad.
- `_adjust_difficulty` (method, `app.py:205`) `def _adjust_difficulty(self, avg_performance)` - Ajusta dificultad basada en rendimiento.
- `AbsoluteZeroCmd` (class, `app.py:217`) `class AbsoluteZeroCmd(Cmd)` - Sistema Absolute Zero mejorado con características avanzadas.
- `__init__` (method, `app.py:220`) `def __init__(self)`
- `_initialize_with_seed` (method, `app.py:237`) `def _initialize_with_seed(self)` - Inicializa con triplete semilla mejorado.
- `do_analyze` (method, `app.py:258`) `def do_analyze(self, args)` - Analiza código con opciones avanzadas.
- `do_stats` (method, `app.py:267`) `def do_stats(self, arg)` - Muestra estadísticas del sistema.
- `do_curriculum` (method, `app.py:282`) `def do_curriculum(self, arg)` - Muestra estado del curriculum learning.
- `do_save_memory` (method, `app.py:289`) `def do_save_memory(self, arg)` - Guarda manualmente el banco de memoria.
- `_analyze_sequential` (method, `app.py:296`) `def _analyze_sequential(self, directory, iterations)` - Análisis secuencial mejorado.
- `_analyze_parallel` (method, `app.py:309`) `def _analyze_parallel(self, directory, iterations)` - Análisis paralelo con threading.
- `_get_code_files` (method, `app.py:326`) `def _get_code_files(self, directory)` - Obtiene lista de archivos de código.
- `_analyze_code_file` (method, `app.py:335`) `def _analyze_code_file(self, file_path, iterations)` - Análisis mejorado de archivo de código.
- `_absolute_zero_loop` (method, `app.py:346`) `def _absolute_zero_loop(self, code_content, file_path, iterations)` - Ciclo principal mejorado con métricas.

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 0
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- No cross-community bridges recorded. This community is self-contained.

## Risks

- [taint medium] `app.py` -> `app.py` via `requests` (0 hops)

## Open Questions

- Why do 1 file(s) lack file-level docs (e.g. `install.sh`)? What purpose do they serve?
- Is the dangerous import `requests` in `app.py` still required, or can it be isolated?
- What would break if the most connected file in root changed?
- Should root be split, given cohesion 1.00?

## Sources

- `app.py`
- `install.sh`
