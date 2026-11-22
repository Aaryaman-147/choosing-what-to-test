
import sys, runpy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv = ['run_test.py', str(ROOT/'data/bace.csv')]
runpy.run_path(str(ROOT/'scripts'/'run_test.py'), run_name='__main__')
