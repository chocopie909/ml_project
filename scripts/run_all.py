from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mlproject.experiments import run_all
if __name__=='__main__':
    for name,table in run_all().items():
        print('\n'+name);print(table.to_string(index=False))
