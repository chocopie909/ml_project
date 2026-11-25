"""Full gate: tests, all experiments, then all notebooks in clean kernels."""
from pathlib import Path
import subprocess,sys,json,platform,time
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    start=time.perf_counter()
    commands=[[sys.executable,'-m','pytest','-q','--junitxml=reports/tests.xml'],
              [sys.executable,'scripts/run_all.py'],[sys.executable,'scripts/execute_notebooks.py']]
    for command in commands:subprocess.run(command,cwd=ROOT,check=True)
    (ROOT/'reports/verification.json').write_text(json.dumps({'status':'passed','python':platform.python_version(),'platform':platform.system(),'checks':['pytest','five_experiments','six_notebooks_clean_kernels'],'seconds':round(time.perf_counter()-start,2)},indent=2),encoding='utf8')
    print('FULL VERIFICATION PASSED')
