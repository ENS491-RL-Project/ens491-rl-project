import json
from pathlib import Path
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--out-dir', type=Path, default=Path('output/training_parity_audit'))
p=parser.parse_args().out_dir
rows=[]
for task,arm in [('unlock','goal'),('solve','goal'),('solve','strict')]:
 for seed in [42,43,44]:
  a=json.loads((p/f'{task}_{arm}_{seed}_old.json').read_text())
  b=json.loads((p/f'{task}_{arm}_{seed}_new.json').read_text())
  assert len(a['transitions'])==len(b['transitions'])
  keys=['before','mask','action','after','reward','term','trunc']
  differences=[(i,x,y) for i,(x,y) in enumerate(zip(a['transitions'],b['transitions'])) if x!=y]
  row={'task':task,'arm':arm,'seed':seed,'weights_equal':a['weights']==b['weights'],
       'all_updates_equal':a['updates']==b['updates'],
       'learner_transition_differences':sum(any(x[k]!=y[k] for k in keys) for x,y in zip(a['transitions'],b['transitions'])),
       'different_fields':sorted({k for _,x,y in differences for k in x if x[k]!=y[k]}),
       'changed_duration_decisions':sum(x['k']!=y['k'] for x,y in zip(a['transitions'],b['transitions'])),
       'nested_mask_changed_decisions':sum(x['nested_masks']!=y['nested_masks'] for x,y in zip(a['transitions'],b['transitions'])),
       'old_env_steps':a['report']['env_steps_own'],'new_env_steps':b['report']['env_steps_own'],
       'actual_train_calls':b['actual_train_calls'],'reported_policy_updates':b['report']['policy_updates'],
       'decisions':b['report']['decisions_trained']}
  rows.append(row)
(p/'summary.json').write_text(json.dumps(rows,indent=2))
for row in rows: print(json.dumps(row))
