
import sys, runpy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv = ['resolve_tie.py', str(ROOT/'data/bace.csv')]
runpy.run_path(str(ROOT/'scripts'/'resolve_tie.py'), run_name='__main__')
