import json, sys
from pathlib import Path
sys.path.insert(0, 'C:/Users/aarya/sm03/pkg_sm03')
from sm03.src import data
ROOT = Path('C:/Users/aarya/sm03/pkg_sm03')
for seed in (42,43,44):
    p = ROOT/f'outputs/test_ledgers/test_ucb_diverse_b05_l03_seed{seed}.json'
    d = json.loads(p.read_text())
    init = d['rounds'][0]['requested_ids']
    df = data.load_source(ROOT/'data/bace.csv')
    pools = data.scaffold_fallback_pools(df, seed=0)
    expected = data.initial_selection(list(pools.test), seed, 40, 'plain')
    print(f'seed{seed}: ledger initial-40 == SHA256 plain -> {set(init) == set(expected)}')
# quote one batch round from seed 42
d = json.loads((ROOT/'outputs/test_ledgers/test_ucb_diverse_b05_l03_seed42.json').read_text())
b = d['rounds'][1]
print('batch1 ids[:3]:', b['requested_ids'][:3])
print('batch1 pred keys == ids:', set(b['predictions_before_reveal']) == set(b['requested_ids']))
print('batch1 sample pred:', {k: round(b['predictions_before_reveal'][k],4) for k in b['requested_ids'][:2]})
print('batch1 sample label:', {k: b['revealed_labels'][k] for k in b['requested_ids'][:2]})
