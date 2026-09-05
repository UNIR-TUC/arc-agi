"""Durable per-task results for crash-resumable ARC runs."""

import datetime
import json
import os


_LOGGER_FIELDS = (
    'solution_contributions_log',
    'solution_picks_history',
)
_RECOVERY_SCHEMA_VERSION = 1
_SEED_PARTIAL_SCHEMA_VERSION = 1
_RECOVERY_STATES = frozenset((
    'retry_eager',
    'quarantined',
    'recovered_eager',
))
_MAX_FAILURE_HISTORY = 20


def safe_task_name(task_name):
    """Reject anything that is not a plain ARC task id before it reaches a path."""
    if not task_name or not all(ch.isalnum() or ch in '-_' for ch in task_name):
        raise ValueError(f'unsafe task name for a file path: {task_name!r}')
    return task_name


def partial_dir(split, state_dir=None):
    return os.path.join(state_dir or '.partial', split)


def recovery_path(split):
    return os.path.join(partial_dir(split), '.task_recovery.json')


def _utc_timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec='seconds'
    )


def _fsync_directory(directory):
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_write_json(path, payload):
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    try:
        with open(tmp, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(directory)
    except BaseException:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
        raise


def is_complete_logger(logger_data):
    return (
        isinstance(logger_data, dict)
        and all(field in logger_data for field in _LOGGER_FIELDS)
    )


def save_task_partial(split, task_name, n_steps, solution, logger_data,
                      state_dir=None):
    """Atomically persist one complete Phase-2 task result.

    The function is intentionally strict: returning success without a durable
    solution would let a long-running scheduler silently lose completed work.
    """
    if not solution:
        raise ValueError(f'cannot persist {task_name}: solution is empty')
    if not is_complete_logger(logger_data):
        raise ValueError(f'cannot persist {task_name}: logger data is incomplete')

    directory = partial_dir(split, state_dir)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f'{safe_task_name(task_name)}.json')
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as handle:
            json.dump({
                'n_steps': n_steps,
                'solution': solution,
                'logger': logger_data,
            }, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
        raise
    return path


def load_task_partials(split, task_names, n_steps, state_dir=None):
    """Return complete matching solutions and loggers from an earlier run."""
    solutions = {}
    loggers = {}
    directory = partial_dir(split, state_dir)
    if not os.path.isdir(directory):
        return solutions, loggers

    for task_name in task_names:
        try:
            path = os.path.join(directory, f'{safe_task_name(task_name)}.json')
            with open(path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
        except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
            continue

        if not isinstance(payload, dict):
            continue
        solution = payload.get('solution')
        logger_data = payload.get('logger')
        if (
            payload.get('n_steps') != n_steps
            or not isinstance(solution, list)
            or not solution
            or not is_complete_logger(logger_data)
        ):
            continue
        solutions[task_name] = solution
        loggers[task_name] = logger_data
    return solutions, loggers


def seed_partial_path(split, task_name, seed, fingerprint, state_dir=None):
    safe_task_name(task_name)
    safe_task_name(fingerprint)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    return os.path.join(
        partial_dir(split, state_dir),
        'eje_b',
        fingerprint,
        task_name,
        f'seed_{seed}.json',
    )


def save_seed_partial(split, task_name, n_steps, seed, fingerprint,
                      solution, logger_data, state_dir=None):
    """Atomically persist one complete Eje B seed job."""
    if not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError('n_steps must be a positive integer')
    if not solution:
        raise ValueError(f'cannot persist {task_name} seed {seed}: solution is empty')
    if not is_complete_logger(logger_data):
        raise ValueError(
            f'cannot persist {task_name} seed {seed}: logger data is incomplete'
        )
    path = seed_partial_path(
        split, task_name, seed, fingerprint, state_dir
    )
    _atomic_write_json(path, {
        'schema_version': _SEED_PARTIAL_SCHEMA_VERSION,
        'fingerprint': fingerprint,
        'task_name': task_name,
        'seed': seed,
        'n_steps': n_steps,
        'solution': solution,
        'logger': logger_data,
    })
    return path


def load_seed_partials(split, task_names, n_steps, seeds, fingerprint,
                       state_dir=None):
    """Return matching Eje B results keyed by ``(task_name, seed)``."""
    solutions = {}
    loggers = {}
    for task_name in task_names:
        for seed in seeds:
            try:
                path = seed_partial_path(
                    split, task_name, seed, fingerprint, state_dir
                )
                with open(path, encoding='utf-8') as handle:
                    payload = json.load(handle)
            except (FileNotFoundError, OSError, ValueError, TypeError,
                    json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            solution = payload.get('solution')
            logger_data = payload.get('logger')
            if (
                payload.get('schema_version') != _SEED_PARTIAL_SCHEMA_VERSION
                or payload.get('fingerprint') != fingerprint
                or payload.get('task_name') != task_name
                or payload.get('seed') != seed
                or payload.get('n_steps') != n_steps
                or not isinstance(solution, list)
                or not solution
                or not is_complete_logger(logger_data)
            ):
                continue
            key = (task_name, seed)
            solutions[key] = solution
            loggers[key] = logger_data
    return solutions, loggers


def _empty_recovery_manifest():
    return {
        'schema_version': _RECOVERY_SCHEMA_VERSION,
        'iterations': {},
    }


def load_task_recovery_manifest(split):
    """Load and validate durable task recovery state for one split."""
    path = recovery_path(split)
    try:
        with open(path, encoding='utf-8') as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return _empty_recovery_manifest()
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'invalid task recovery manifest: {path}') from exc

    if (
        not isinstance(payload, dict)
        or payload.get('schema_version') != _RECOVERY_SCHEMA_VERSION
        or not isinstance(payload.get('iterations'), dict)
    ):
        raise ValueError(f'invalid task recovery manifest: {path}')

    for steps, task_entries in payload['iterations'].items():
        if not isinstance(steps, str) or not steps.isdigit():
            raise ValueError(f'invalid task recovery manifest: {path}')
        if not isinstance(task_entries, dict):
            raise ValueError(f'invalid task recovery manifest: {path}')
        for task_name, entry in task_entries.items():
            try:
                safe_task_name(task_name)
            except ValueError as exc:
                raise ValueError(f'invalid task recovery manifest: {path}') from exc
            if (
                not isinstance(entry, dict)
                or entry.get('state') not in _RECOVERY_STATES
                or not isinstance(entry.get('failures', []), list)
            ):
                raise ValueError(f'invalid task recovery manifest: {path}')
    return payload


def load_task_recovery(split, n_steps):
    """Return recovery entries scoped to the requested iteration count."""
    manifest = load_task_recovery_manifest(split)
    return dict(manifest['iterations'].get(str(n_steps), {}))


def update_task_recovery(split, task_name, n_steps, state, failure=None):
    """Atomically transition one task's recovery state."""
    safe_task_name(task_name)
    if state not in _RECOVERY_STATES:
        raise ValueError(f'unknown task recovery state: {state!r}')
    if not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError('n_steps must be a positive integer')
    if failure is not None and not isinstance(failure, dict):
        raise ValueError('failure must be a dictionary or None')

    manifest = load_task_recovery_manifest(split)
    task_entries = manifest['iterations'].setdefault(str(n_steps), {})
    previous = task_entries.get(task_name, {})
    failures = list(previous.get('failures', []))
    if failure is not None:
        failures.append(dict(failure))
        failures = failures[-_MAX_FAILURE_HISTORY:]
    task_entries[task_name] = {
        'state': state,
        'updated_at': _utc_timestamp(),
        'failures': failures,
    }
    _atomic_write_json(recovery_path(split), manifest)
    return dict(task_entries[task_name])