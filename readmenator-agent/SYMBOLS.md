# Symbols

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
| `AbsoluteZeroCmd` | class | `app.py:217` | `class AbsoluteZeroCmd(Cmd)` |
| `AbsoluteZeroCmd` | class | `app.py:1064` | `class AbsoluteZeroCmd(AbsoluteZeroCmd)` |
| `AdvancedAnalytics` | class | `app.py:984` | `class AdvancedAnalytics` |
| `CurriculumLearning` | class | `app.py:185` | `class CurriculumLearning` |
| `MemoryBank` | class | `app.py:93` | `class MemoryBank` |
| `MetaLearning` | class | `app.py:871` | `class MetaLearning` |
| `SelfPlayTraining` | class | `app.py:744` | `class SelfPlayTraining` |
| `Task` | class | `app.py:62` | `class Task` |
| `__init__` | method | `app.py:96` | `def __init__(self, persist_path)` |
| `__init__` | method | `app.py:188` | `def __init__(self)` |
| `__init__` | method | `app.py:220` | `def __init__(self)` |
| `__init__` | method | `app.py:747` | `def __init__(self, memory_bank, cmd_instance)` |
| `__init__` | method | `app.py:874` | `def __init__(self)` |
| `__init__` | method | `app.py:987` | `def __init__(self)` |
| `__init__` | method | `app.py:1067` | `def __init__(self)` |
| `__post_init__` | method | `app.py:75` | `def __post_init__(self)` |
| `_absolute_zero_loop` | method | `app.py:346` | `def _absolute_zero_loop(self, code_content, file_path, iterations)` |
| `_absolute_zero_loop` | method | `app.py:1199` | `def _absolute_zero_loop(self, code_content, file_path, iterations)` |
| `_adjust_difficulty` | method | `app.py:205` | `def _adjust_difficulty(self, avg_performance)` |
| `_analyze_code_file` | method | `app.py:335` | `def _analyze_code_file(self, file_path, iterations)` |
| `_analyze_parallel` | method | `app.py:309` | `def _analyze_parallel(self, directory, iterations)` |
| `_analyze_sequential` | method | `app.py:296` | `def _analyze_sequential(self, directory, iterations)` |
| `_apply_hyperparameters` | method | `app.py:927` | `def _apply_hyperparameters(self, cmd_instance, hyperparams)` |
| `_build_adaptive_propose_prompt` | method | `app.py:393` | `def _build_adaptive_propose_prompt(self, task_type, code_content, ref_tasks, difficulty)` |
| `_build_solve_prompt_improved` | method | `app.py:577` | `def _build_solve_prompt_improved(self, task)` |
| `_calculate_average_reward` | method | `app.py:704` | `def _calculate_average_reward(self, solutions)` |
| `_calculate_reward_improved` | method | `app.py:634` | `def _calculate_reward_improved(self, task, solution)` |
| `_compute_hash` | method | `app.py:81` | `def _compute_hash(self)` |
| `_create_benchmark_tasks` | method | `app.py:1171` | `def _create_benchmark_tasks(self)` |
| `_display_optimization_summary` | method | `app.py:976` | `def _display_optimization_summary(self)` |
| `_display_tournament_summary` | method | `app.py:857` | `def _display_tournament_summary(self)` |
| `_evaluate_baseline_task` | method | `app.py:831` | `def _evaluate_baseline_task(self, task)` |
| `_evaluate_performance` | method | `app.py:936` | `def _evaluate_performance(self, cmd_instance)` |
| `_evaluate_single_task` | method | `app.py:821` | `def _evaluate_single_task(self, task)` |
| `_evaluate_tournament_round` | method | `app.py:797` | `def _evaluate_tournament_round(self, tasks)` |
| `_extract_function_name` | method | `app.py:533` | `def _extract_function_name(self, program)` |
| `_extract_solution_improved` | method | `app.py:617` | `def _extract_solution_improved(self, response, task_type)` |
| `_generate_hyperparameters` | method | `app.py:917` | `def _generate_hyperparameters(self)` |
| `_generate_test_tasks` | method | `app.py:952` | `def _generate_test_tasks(self)` |
| `_get_code_files` | method | `app.py:326` | `def _get_code_files(self, directory)` |
| `_initialize_with_seed` | method | `app.py:237` | `def _initialize_with_seed(self)` |
| `_is_executable` | method | `app.py:482` | `def _is_executable(self, program)` |
| `_is_too_similar` | method | `app.py:120` | `def _is_too_similar(self, new_task)` |
| `_maintain_buffer_size` | method | `app.py:149` | `def _maintain_buffer_size(self, task_type)` |
| `_propose_tasks_adaptive` | method | `app.py:364` | `def _propose_tasks_adaptive(self, code_content, file_path)` |
| `_query_deepseek_improved` | method | `app.py:423` | `def _query_deepseek_improved(self, prompt)` |
| `_select_tournament_tasks` | method | `app.py:778` | `def _select_tournament_tasks(self, n_tasks)` |
| `_simple_embedding` | method | `app.py:137` | `def _simple_embedding(self, program)` |
| `_solve_tasks_improved` | method | `app.py:538` | `def _solve_tasks_improved(self, tasks, file_path)` |
| `_update_elo_ratings` | method | `app.py:841` | `def _update_elo_ratings(self, results)` |
| `_update_model_improved` | method | `app.py:712` | `def _update_model_improved(self, tasks, solutions)` |
| `_validate_abduction_task` | method | `app.py:504` | `def _validate_abduction_task(self, task)` |
| `_validate_deduction_task` | method | `app.py:490` | `def _validate_deduction_task(self, task)` |
| `_validate_induction_task` | method | `app.py:516` | `def _validate_induction_task(self, task)` |
| `_validate_task_improved` | method | `app.py:457` | `def _validate_task_improved(self, task)` |
| `_verify_abduction_solution` | method | `app.py:669` | `def _verify_abduction_solution(self, task, solution)` |
| `_verify_deduction_solution` | method | `app.py:657` | `def _verify_deduction_solution(self, task, solution)` |
| `_verify_induction_solution` | method | `app.py:681` | `def _verify_induction_solution(self, task, solution)` |
| `add_task` | method | `app.py:103` | `def add_task(self, task)` |
| `do_analytics` | method | `app.py:1092` | `def do_analytics(self, arg)` |
| `do_analyze` | method | `app.py:258` | `def do_analyze(self, args)` |
| `do_benchmark` | method | `app.py:1132` | `def do_benchmark(self, arg)` |
| `do_curriculum` | method | `app.py:282` | `def do_curriculum(self, arg)` |
| `do_export_tasks` | method | `app.py:1100` | `def do_export_tasks(self, arg)` |
| `do_import_tasks` | method | `app.py:1111` | `def do_import_tasks(self, arg)` |
| `do_optimize` | method | `app.py:1084` | `def do_optimize(self, arg)` |
| `do_save_memory` | method | `app.py:289` | `def do_save_memory(self, arg)` |
| `do_stats` | method | `app.py:267` | `def do_stats(self, arg)` |
| `do_tournament` | method | `app.py:1076` | `def do_tournament(self, arg)` |
| `generate_report` | method | `app.py:1027` | `def generate_report(self)` |
| `get_reference_tasks` | method | `app.py:156` | `def get_reference_tasks(self, task_type, n)` |
| `load_from_disk` | method | `app.py:175` | `def load_from_disk(self)` |
| `optimize_hyperparameters` | method | `app.py:887` | `def optimize_hyperparameters(self, cmd_instance, iterations)` |
| `run_tournament` | method | `app.py:753` | `def run_tournament(self, rounds)` |
| `save_analytics` | method | `app.py:1056` | `def save_analytics(self, filepath)` |
| `save_to_disk` | method | `app.py:167` | `def save_to_disk(self)` |
| `success_rate` | method | `app.py:87` | `def success_rate(self)` |
| `to_dict` | method | `app.py:90` | `def to_dict(self)` |
| `track_error` | method | `app.py:1020` | `def track_error(self, error_type, context)` |
| `track_learning_velocity` | method | `app.py:1013` | `def track_learning_velocity(self, success_rate)` |
| `track_performance` | method | `app.py:1005` | `def track_performance(self, reward, task_type)` |
| `track_task_creation` | method | `app.py:996` | `def track_task_creation(self, task)` |
| `update_performance` | method | `app.py:198` | `def update_performance(self, reward)` |
