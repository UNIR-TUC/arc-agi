"""Select ARC tasks from a validated historical result list."""

import argparse
import json
import re


_RESULT_LINE = re.compile(
    r'^\s*[^|]+\|\s*(?P<split>[^|]+?)\s*\|\s*'
    r'\[\s*\d+\s*\]\s*(?P<task_id>[A-Za-z0-9_-]+)\s*\|\s*'
    r'guess@[12]\s*$'
)


def _completed_task_ids(results_path, split):
    completed = []
    seen = set()
    with open(results_path, encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            match = _RESULT_LINE.fullmatch(line.rstrip('\n'))
            if match is None:
                raise ValueError(
                    f'invalid result line at {results_path}:{line_number}'
                )
            result_split = match.group('split').strip()
            if result_split != split:
                raise ValueError(
                    f'result line at {results_path}:{line_number} belongs '
                    f'to split {result_split!r}, expected {split!r}'
                )
            task_id = match.group('task_id')
            if task_id in seen:
                raise ValueError(
                    f'duplicate task id {task_id!r} at '
                    f'{results_path}:{line_number}'
                )
            seen.add(task_id)
            completed.append(task_id)
    return completed


def select_unresolved_task_ids(
    challenges_path,
    results_path,
    split='training',
    expected_excluded=None,
    expected_remaining=None,
):
    """Return unresolved task IDs in the challenge file's original order."""
    with open(challenges_path, encoding='utf-8') as handle:
        challenges = json.load(handle)
    if not isinstance(challenges, dict):
        raise ValueError(f'challenges file is not a JSON object: {challenges_path}')

    known_ids = list(challenges)
    known = set(known_ids)
    completed = _completed_task_ids(results_path, split)
    unknown = sorted(set(completed) - known)
    if unknown:
        raise ValueError(
            'historical results contain unknown task id(s): '
            + ', '.join(unknown)
        )
    if expected_excluded is not None and len(completed) != expected_excluded:
        raise ValueError(
            f'expected {expected_excluded} excluded tasks, found {len(completed)}'
        )

    completed_set = set(completed)
    unresolved = [task_id for task_id in known_ids if task_id not in completed_set]
    if expected_remaining is not None and len(unresolved) != expected_remaining:
        raise ValueError(
            f'expected {expected_remaining} unresolved tasks, found {len(unresolved)}'
        )
    return unresolved


def main():
    parser = argparse.ArgumentParser(
        description='Print unresolved ARC task IDs in dataset order.'
    )
    parser.add_argument('--challenges', required=True, metavar='PATH')
    parser.add_argument('--results', required=True, metavar='PATH')
    parser.add_argument('--split', default='training')
    parser.add_argument('--expected-excluded', type=int, default=None)
    parser.add_argument('--expected-remaining', type=int, default=None)
    args = parser.parse_args()

    try:
        task_ids = select_unresolved_task_ids(
            args.challenges,
            args.results,
            split=args.split,
            expected_excluded=args.expected_excluded,
            expected_remaining=args.expected_remaining,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(','.join(task_ids))


if __name__ == '__main__':
    main()