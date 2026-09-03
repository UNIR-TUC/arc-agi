import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class FullSplitRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        source_root = Path(__file__).resolve().parent
        for name in ('run_full_split.sh', 'run_all_full.sh', 'run_tracking.py'):
            shutil.copy2(source_root / name, self.root / name)

        (self.root / 'dataset').mkdir()
        for split in ('training', 'evaluation', 'test'):
          (self.root / 'dataset' / f'arc-agi_{split}_challenges.json').write_text(
            '{}', encoding='utf-8',
          )
        (self.root / 'cache').mkdir()
        self.fake_bin = self.root / 'fake-bin'
        self.fake_bin.mkdir()
        self._write_fake_commands()

    def _write_executable(self, name, contents):
        path = self.fake_bin / name
        path.write_text(contents, encoding='utf-8')
        path.chmod(0o755)
        return path

    def _write_fake_commands(self):
        self._write_executable('findmnt', '''#!/usr/bin/env bash
case "${*: -1}" in
  TARGET) echo "$CACHE_MOUNT" ;;
  SOURCE) echo "/dev/fake-cache-partition" ;;
  FSTYPE) echo "ext4" ;;
  OPTIONS) echo "rw,relatime" ;;
  UUID) echo "$CACHE_EXPECTED_UUID" ;;
esac
''')
        self._write_executable('lsblk', '''#!/usr/bin/env bash
if [[ "$*" == *"PKNAME"* ]]; then
  echo "fake-cache-device"
else
  echo "$CACHE_EXPECTED_SERIAL"
fi
''')
        self._write_executable('df', '''#!/usr/bin/env bash
if [[ "$*" == *"--output=iavail"* ]]; then
  printf 'IAvail\n2000000\n'
else
  printf 'Avail\n1000G\n'
fi
''')

        real_python = Path(os.sys.executable)
        self.python_wrapper = self._write_executable(
            'fake-python', f'''#!/usr/bin/env bash
if [[ "$1" == "run_tracking.py" ]]; then
  exec "{real_python}" "$@"
fi
if [[ "$1" != "-u" || "$2" != "parallel_train.py" ]]; then
  echo "unexpected fake-python arguments: $*" >&2
  exit 99
fi
printf '%s\n' "$*" >> "$FAKE_ARGS_FILE"
count=0
if [[ -f "$FAKE_STATE_FILE" ]]; then
  count="$(<"$FAKE_STATE_FILE")"
fi
count=$(( count + 1 ))
printf '%s\n' "$count" > "$FAKE_STATE_FILE"
IFS=',' read -r -a codes <<< "$FAKE_EXIT_SEQUENCE"
index=$(( count - 1 ))
code="${{codes[$index]:-${{codes[-1]}}}}"
if (( code == 0 )); then
  split=""
  while (( $# )); do
    if [[ "$1" == "--split" ]]; then
      split="$2"
      break
    fi
    shift
  done
  "{real_python}" -c 'import json, sys; json.dump({{"elapsed_s": 1.0, "phase1_s": 0.0, "phase2_s": 1.0}}, open(f"run_metadata_{{sys.argv[1]}}.json", "w"))' "$split"
  printf '{{}}\n' > "submission_${{split}}.json"
  : > "predictions_${{split}}.npz"
fi
exit "$code"
''',
        )

    def _environment(self, exit_sequence, max_attempts=2):
        state_file = self.root / 'attempt-count'
        env = os.environ.copy()
        env.update({
            'PATH': f'{self.fake_bin}:{env["PATH"]}',
            'PYTHON_BIN': str(self.python_wrapper),
            'FAKE_STATE_FILE': str(state_file),
            'FAKE_EXIT_SEQUENCE': exit_sequence,
            'FAKE_ARGS_FILE': str(self.root / 'python-arguments'),
            'MAX_ATTEMPTS': str(max_attempts),
            'RETRY_DELAY_S': '0',
            'CACHE_MOUNT': str(self.root / 'cache'),
            'CACHE_EXPECTED_UUID': 'fake-uuid',
            'CACHE_EXPECTED_SERIAL': 'fake-serial',
            'INDUCTOR_CACHE_DIR': str(self.root / 'cache' / '.inductor_cache'),
            'CACHE_WARN_FREE_GB': '400',
            'CACHE_MIN_FREE_GB': '50',
            'CACHE_MIN_FREE_INODES': '1000000',
            'NO_COLOR': '1',
        })
        return env, state_file

    def _run(self, exit_sequence, max_attempts=2):
        env, state_file = self._environment(exit_sequence, max_attempts)
        result = subprocess.run(
            ['bash', 'run_full_split.sh', 'training'],
            cwd=self.root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        count = int(state_file.read_text()) if state_file.exists() else 0
        summary = json.loads(
            (self.root / 'run_summary_training.json').read_text(encoding='utf-8')
        )
        log_path = self.root / summary['log_path']
        log_contents = log_path.read_text(encoding='utf-8')
        return result, count, summary, log_contents

    def test_failed_attempt_is_retried_and_success_is_summarized(self):
        result, count, summary, log_contents = self._run('1,0')

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(count, 2)
        self.assertEqual(summary['status'], 'success')
        self.assertEqual(summary['attempt_count'], 2)
        self.assertEqual(summary['retry_count'], 1)
        self.assertIn('exit=1', log_contents)
        self.assertIn('retrying in 0s', log_contents)
        metadata = json.loads(
            (self.root / 'run_metadata_training.json').read_text()
        )
        self.assertEqual(metadata['runner']['attempt_count'], 2)

    def test_recovery_environment_is_forwarded_to_python(self):
        env, _ = self._environment('0')
        env.update({
            'ACCEL_PRESET': 'baseline',
            'EAGER_TASKS': 'fcb5c309',
            'RETRY_QUARANTINED_TASKS': '007bbfb7',
        })

        result = subprocess.run(
            ['bash', 'run_full_split.sh', 'training'],
            cwd=self.root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = (self.root / 'python-arguments').read_text()
        self.assertIn('--accel-preset baseline', arguments)
        self.assertIn('--recover-task-failures', arguments)
        self.assertIn('--eager-tasks fcb5c309', arguments)
        self.assertIn('--retry-quarantined 007bbfb7', arguments)

    def test_quarantined_retry_flag_is_only_sent_on_first_attempt(self):
        env, state_file = self._environment('1,0')
        env['RETRY_QUARANTINED_TASKS'] = 'fcb5c309'

        result = subprocess.run(
            ['bash', 'run_full_split.sh', 'training'],
            cwd=self.root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(int(state_file.read_text()), 2)
        argument_lines = (
            self.root / 'python-arguments'
        ).read_text().splitlines()
        self.assertIn('--retry-quarantined fcb5c309', argument_lines[0])
        self.assertNotIn('--retry-quarantined', argument_lines[1])

    def test_exit_137_is_diagnosed_and_retried(self):
        result, count, summary, log_contents = self._run('137,0')

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(count, 2)
        self.assertEqual(summary['attempts'][0]['outcome'], 'killed')
        self.assertIn('SIGKILL', log_contents)
        self.assertIn('may be the OS OOM killer', log_contents)

    def test_operator_interrupt_is_not_retried(self):
        result, count, summary, _ = self._run('130,0')

        self.assertEqual(result.returncode, 130)
        self.assertEqual(count, 1)
        self.assertEqual(summary['status'], 'interrupted')
        self.assertEqual(summary['attempt_count'], 1)

    def test_exhausted_attempts_return_failure_with_summary(self):
        result, count, summary, _ = self._run('1,1')

        self.assertEqual(result.returncode, 1)
        self.assertEqual(count, 2)
        self.assertEqual(summary['status'], 'failed')
        self.assertEqual(summary['attempt_count'], 2)

    def test_all_runner_aggregates_three_splits_into_one_campaign(self):
        env, state_file = self._environment('0')
        env['CAMPAIGN_ID'] = 'integration-campaign'

        result = subprocess.run(
            ['bash', 'run_all_full.sh'],
            cwd=self.root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(int(state_file.read_text()), 3)
        summary = json.loads(
            (self.root / 'campaign_summary.json').read_text(encoding='utf-8')
        )
        self.assertEqual(summary['campaign_id'], 'integration-campaign')
        self.assertEqual(summary['status'], 'success')
        self.assertEqual(summary['split_count'], 3)
        self.assertEqual(summary['attempt_count'], 3)
        self.assertEqual(
            [item['split'] for item in summary['splits']],
            ['training', 'evaluation', 'test'],
        )
        self.assertIn(
            'Splits run   : training, evaluation, test',
            (self.root / 'timing_result.txt').read_text(encoding='utf-8'),
        )


if __name__ == '__main__':
    unittest.main()