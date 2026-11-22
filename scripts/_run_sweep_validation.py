
import sys, runpy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv = ['sweep_validation.py', str(ROOT/'data/bace.csv')]
runpy.run_path(str(ROOT/'scripts'/'sweep_validation.py'), run_name='__main__')
