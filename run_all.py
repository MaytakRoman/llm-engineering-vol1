"""Run all book listings from the project root and retain complete logs."""
from pathlib import Path
import json
import os
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def main():
    logs = ROOT / 'execution_logs'
    logs.mkdir(exist_ok=True)
    scripts = [ROOT / 'case_experiment.py'] + sorted(ROOT.glob('ch*_listing*.py'))
    report = {'python': platform.python_version(), 'platform': platform.platform(), 'runs': []}
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'PYTHONIOENCODING': 'utf-8'}
    for script in scripts:
        started = time.perf_counter()
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env,
                                capture_output=True, text=True, encoding='utf-8')
        (logs / (script.stem + '.txt')).write_text(result.stdout + result.stderr, encoding='utf-8')
        report['runs'].append({'file': script.name, 'returncode': result.returncode,
                              'seconds': round(time.perf_counter() - started, 4),
                              'stdout': result.stdout, 'stderr': result.stderr})
        print(script.name, result.returncode, flush=True)
    report['passed'] = sum(row['returncode'] == 0 for row in report['runs'])
    report['total'] = len(scripts)
    (ROOT / 'execution_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return int(report['passed'] != report['total'])

if __name__ == '__main__':
    sys.exit(main())
