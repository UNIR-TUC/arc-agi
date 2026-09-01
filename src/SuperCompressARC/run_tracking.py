"""Durable timing and outcome records for retrying split runners."""

import argparse
import datetime
import json
import os


def _timestamp(epoch):
    return datetime.datetime.fromtimestamp(
        float(epoch), datetime.timezone.utc,
    ).astimezone().isoformat(timespec='seconds')


def _fsync_directory(path):
    directory_fd = os.open(path or '.', os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_write_json(path, payload):
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


def append_event(path, event):
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    line = json.dumps(event, separators=(',', ':')) + '\n'
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(directory)


def record_attempt_start(events_path, run_id, campaign_id, split, attempt,
                         started_at, log_path):
    append_event(events_path, {
        'event': 'attempt_started',
        'run_id': run_id,
        'campaign_id': campaign_id,
        'split': split,
        'attempt': int(attempt),
        'at_epoch': float(started_at),
        'at': _timestamp(started_at),
        'log_path': log_path,
    })


def _outcome(exit_code):
    if exit_code == 0:
        return 'success'
    if exit_code in (130, 143):
        return 'interrupted'
    if exit_code == 137:
        return 'killed'
    return 'failed'


def record_attempt_end(events_path, run_id, campaign_id, split, attempt,
                       ended_at, exit_code):
    append_event(events_path, {
        'event': 'attempt_finished',
        'run_id': run_id,
        'campaign_id': campaign_id,
        'split': split,
        'attempt': int(attempt),
        'at_epoch': float(ended_at),
        'at': _timestamp(ended_at),
        'exit_code': int(exit_code),
        'outcome': _outcome(int(exit_code)),
    })


def load_events(path):
    if not os.path.exists(path):
        return []
    events = []
    with open(path, encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f'invalid tracking event at {path}:{line_number}'
                ) from exc
    return events


def build_split_summary(events, run_id, campaign_id, split, started_at,
                        ended_at, status, log_path):
    attempts = {}
    for event in events:
        if (
            event.get('run_id') != run_id
            or event.get('campaign_id') != campaign_id
            or event.get('split') != split
        ):
            continue
        attempt = int(event['attempt'])
        record = attempts.setdefault(attempt, {'attempt': attempt})
        if event.get('event') == 'attempt_started':
            if 'started_at_epoch' in record:
                raise ValueError(f'duplicate start event for attempt {attempt}')
            record.update({
                'started_at_epoch': float(event['at_epoch']),
                'started_at': event['at'],
                'log_path': event.get('log_path'),
            })
        elif event.get('event') == 'attempt_finished':
            if 'ended_at_epoch' in record:
                raise ValueError(f'duplicate end event for attempt {attempt}')
            record.update({
                'ended_at_epoch': float(event['at_epoch']),
                'ended_at': event['at'],
                'exit_code': int(event['exit_code']),
                'outcome': event['outcome'],
            })

    attempt_records = []
    for attempt, record in sorted(attempts.items()):
        if 'started_at_epoch' not in record:
            raise ValueError(f'attempt {attempt} has an end event but no start')
        effective_end = record.get('ended_at_epoch', float(ended_at))
        record['elapsed_s'] = round(
            max(0.0, effective_end - record['started_at_epoch']), 3,
        )
        if 'ended_at_epoch' not in record:
            record.update({
                'ended_at_epoch': float(ended_at),
                'ended_at': _timestamp(ended_at),
                'exit_code': None,
                'outcome': 'interrupted',
            })
        attempt_records.append(record)

    active_s = round(sum(item['elapsed_s'] for item in attempt_records), 3)
    wall_s = round(max(0.0, float(ended_at) - float(started_at)), 3)
    return {
        'schema_version': 1,
        'run_id': run_id,
        'campaign_id': campaign_id,
        'split': split,
        'status': status,
        'started_at_epoch': float(started_at),
        'started_at': _timestamp(started_at),
        'ended_at_epoch': float(ended_at),
        'ended_at': _timestamp(ended_at),
        'attempt_count': len(attempt_records),
        'retry_count': max(0, len(attempt_records) - 1),
        'cumulative_active_s': active_s,
        'total_wall_s': wall_s,
        'retry_wait_and_overhead_s': round(max(0.0, wall_s - active_s), 3),
        'log_path': log_path,
        'degraded': False,
        'quarantined_tasks': [],
        'attempts': attempt_records,
        'outputs': {
            'submission': f'submission_{split}.json',
            'predictions': f'predictions_{split}.npz',
            'run_metadata': f'run_metadata_{split}.json',
        },
    }


def write_split_summary(events_path, summary_path, latest_summary_path,
                        metadata_path, run_id, campaign_id, split,
                        started_at, ended_at, status, log_path):
    summary = build_split_summary(
        load_events(events_path), run_id, campaign_id, split,
        started_at, ended_at, status, log_path,
    )
    metadata = None
    if status == 'success':
        if not metadata_path or not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f'successful run did not write {metadata_path or "run metadata"}'
            )
        if os.path.getmtime(metadata_path) < float(started_at):
            raise ValueError(f'successful run metadata is stale: {metadata_path}')
        with open(metadata_path, encoding='utf-8') as handle:
            metadata = json.load(handle)
        recovery = metadata.get('recovery', {})
        summary['degraded'] = bool(metadata.get('degraded', False))
        summary['quarantined_tasks'] = list(
            recovery.get('quarantined_tasks', [])
        )
        summary['real_result_tasks'] = metadata.get('real_result_tasks')

    atomic_write_json(summary_path, summary)
    if latest_summary_path and latest_summary_path != summary_path:
        atomic_write_json(latest_summary_path, summary)

    if metadata is not None:
        metadata['runner'] = {
            'run_id': run_id,
            'campaign_id': campaign_id,
            'attempt_count': summary['attempt_count'],
            'retry_count': summary['retry_count'],
            'cumulative_active_s': summary['cumulative_active_s'],
            'total_wall_s': summary['total_wall_s'],
            'retry_wait_and_overhead_s': summary['retry_wait_and_overhead_s'],
            'summary_path': summary_path,
        }
        atomic_write_json(metadata_path, metadata)
    return summary


def build_campaign_summary(summary_paths, campaign_id, started_at, ended_at,
                           status):
    splits = []
    for path in summary_paths:
        with open(path, encoding='utf-8') as handle:
            summary = json.load(handle)
        if summary.get('campaign_id') != campaign_id:
            raise ValueError(
                f'{path} belongs to campaign {summary.get("campaign_id")!r}, '
                f'not {campaign_id!r}'
            )
        splits.append({
            'split': summary['split'],
            'status': summary['status'],
            'degraded': bool(summary.get('degraded', False)),
            'quarantined_tasks': list(summary.get('quarantined_tasks', [])),
            'attempt_count': summary['attempt_count'],
            'retry_count': summary['retry_count'],
            'cumulative_active_s': summary['cumulative_active_s'],
            'total_wall_s': summary['total_wall_s'],
            'summary_path': path,
            'outputs': summary['outputs'],
        })

    wall_s = round(max(0.0, float(ended_at) - float(started_at)), 3)
    active_s = round(sum(item['cumulative_active_s'] for item in splits), 3)
    return {
        'schema_version': 1,
        'campaign_id': campaign_id,
        'status': status,
        'degraded': any(item['degraded'] for item in splits),
        'started_at_epoch': float(started_at),
        'started_at': _timestamp(started_at),
        'ended_at_epoch': float(ended_at),
        'ended_at': _timestamp(ended_at),
        'split_count': len(splits),
        'attempt_count': sum(item['attempt_count'] for item in splits),
        'retry_count': sum(item['retry_count'] for item in splits),
        'cumulative_active_s': active_s,
        'total_wall_s': wall_s,
        'wait_and_overhead_s': round(max(0.0, wall_s - active_s), 3),
        'splits': splits,
    }


def write_campaign_summary(summary_paths, output_path, archive_path,
                           timing_path, campaign_id, started_at, ended_at,
                           status):
    summary = build_campaign_summary(
        summary_paths, campaign_id, started_at, ended_at, status,
    )
    atomic_write_json(archive_path, summary)
    if output_path != archive_path:
        atomic_write_json(output_path, summary)
    if timing_path and status == 'success':
        split_names = ', '.join(item['split'] for item in summary['splits'])
        text = (
            f'Campaign ID  : {campaign_id}\n'
            f'Splits run   : {split_names}\n'
            f'Degraded     : {"yes" if summary["degraded"] else "no"}\n'
            f'Attempts     : {summary["attempt_count"]}\n'
            f'Active time  : {summary["cumulative_active_s"]:.1f}s\n'
            f'Total time   : {summary["total_wall_s"]:.1f}s\n'
        )
        directory = os.path.dirname(timing_path) or '.'
        os.makedirs(directory, exist_ok=True)
        tmp = f'{timing_path}.tmp.{os.getpid()}'
        try:
            with open(tmp, 'w', encoding='utf-8') as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, timing_path)
            _fsync_directory(directory)
        except BaseException:
            try:
                os.remove(tmp)
            except FileNotFoundError:
                pass
            raise
    return summary


def _common_attempt_arguments(parser):
    parser.add_argument('--events', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--campaign-id', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--attempt', required=True, type=int)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)

    attempt_start = commands.add_parser('attempt-start')
    _common_attempt_arguments(attempt_start)
    attempt_start.add_argument('--at', required=True, type=float)
    attempt_start.add_argument('--log-path', required=True)

    attempt_end = commands.add_parser('attempt-end')
    _common_attempt_arguments(attempt_end)
    attempt_end.add_argument('--at', required=True, type=float)
    attempt_end.add_argument('--exit-code', required=True, type=int)

    split_summary = commands.add_parser('split-summary')
    split_summary.add_argument('--events', required=True)
    split_summary.add_argument('--summary', required=True)
    split_summary.add_argument('--latest-summary')
    split_summary.add_argument('--metadata')
    split_summary.add_argument('--run-id', required=True)
    split_summary.add_argument('--campaign-id', required=True)
    split_summary.add_argument('--split', required=True)
    split_summary.add_argument('--started-at', required=True, type=float)
    split_summary.add_argument('--ended-at', required=True, type=float)
    split_summary.add_argument('--status', required=True)
    split_summary.add_argument('--log-path', required=True)

    campaign_summary = commands.add_parser('campaign-summary')
    campaign_summary.add_argument('--split-summary', action='append', default=[])
    campaign_summary.add_argument('--output', required=True)
    campaign_summary.add_argument('--archive', required=True)
    campaign_summary.add_argument('--timing-result')
    campaign_summary.add_argument('--campaign-id', required=True)
    campaign_summary.add_argument('--started-at', required=True, type=float)
    campaign_summary.add_argument('--ended-at', required=True, type=float)
    campaign_summary.add_argument('--status', required=True)

    args = parser.parse_args(argv)
    if args.command == 'attempt-start':
        record_attempt_start(
            args.events, args.run_id, args.campaign_id, args.split,
            args.attempt, args.at, args.log_path,
        )
    elif args.command == 'attempt-end':
        record_attempt_end(
            args.events, args.run_id, args.campaign_id, args.split,
            args.attempt, args.at, args.exit_code,
        )
    elif args.command == 'split-summary':
        summary = write_split_summary(
            args.events, args.summary, args.latest_summary, args.metadata,
            args.run_id, args.campaign_id, args.split,
            args.started_at, args.ended_at, args.status, args.log_path,
        )
        print(
            f'[run] timing: {summary["attempt_count"]} attempt(s), '
            f'{summary["cumulative_active_s"]:.1f}s active, '
            f'{summary["total_wall_s"]:.1f}s wall; {args.summary}'
        )
    else:
        summary = write_campaign_summary(
            args.split_summary, args.output, args.archive,
            args.timing_result, args.campaign_id,
            args.started_at, args.ended_at, args.status,
        )
        print(
            f'[run] campaign timing: {summary["attempt_count"]} attempt(s), '
            f'{summary["cumulative_active_s"]:.1f}s active, '
            f'{summary["total_wall_s"]:.1f}s wall; {args.output}'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())