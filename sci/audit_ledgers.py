import json, hashlib
from pathlib import Path
ROOT = Path('C:/Users/aarya/sm03/pkg_sm03')
led = ROOT/'outputs/test_ledgers'
for name in ('ucb_diverse_b05_l03','random','greedy'):
    for seed in (42,43,44):
        p = led/f'test_{name}_seed{seed}.json'
        d = json.loads(p.read_text())
        rounds = d['rounds']
        init = rounds[0]; batches = rounds[1:]
        all_ids = init['requested_ids'] + [i for b in batches for i in b['requested_ids']]
        # SHA256 check: recompute the expected initial-40
        pool = set()
        # reconstruct pool from labels present in ledger metadata? pool size is in d
        # use the recorded initial ids and verify they are the 40 lowest hashes of the test pool
        # test pool = all cids that ever appear + unqueried; simpler: verify internal consistency
        sizes = [len(init['requested_ids'])] + [len(b['requested_ids']) for b in batches]
        dup = len(all_ids) != len(set(all_ids))
        # predictions recorded before reveal: every batch has predictions_before_reveal covering its ids
        pred_ok = all(set(b['predictions_before_reveal']) == set(b['requested_ids']) for b in batches)
        # ledger order: predictions key present and populated
        pred_nonempty = all(len(b['predictions_before_reveal'])==20 for b in batches)
        print(f'{name}/seed{seed}: rounds={len(rounds)} sizes={sizes} total={len(all_ids)} dup={dup} pred_cov={pred_ok} nonempty={pred_nonempty} total_revealed={d["total_revealed"]} pool={d["pool_size"]} batches_used={d["budget"]["batches_used"]}')
