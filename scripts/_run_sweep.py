import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
import runpy
runpy.run_path(str(ROOT/'scripts'/'sweep_validation.py'), run_name='__main__')
