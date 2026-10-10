"""Create a fully synthetic, seeded attack-path benchmark. No enterprise records."""
import numpy as np
import pandas as pd


def generate_dataset(seed=42, n_vulnerabilities=300, n_paths=1250, n_critical=100, n_reachable=80):
    if not (2 <= n_reachable <= n_critical and n_vulnerabilities >= 50):
        raise ValueError('Invalid scenario dimensions')
    rng = np.random.default_rng(seed)
    severity = rng.uniform(3, 10, n_vulnerabilities)
    exploitability = rng.beta(2, 3, n_vulnerabilities)
    exposed = rng.binomial(1, 0.4, n_vulnerabilities)
    asset_criticality = rng.uniform(0.1, 1.0, n_vulnerabilities)
    privilege = rng.binomial(1, 0.35, n_vulnerabilities)
    # Latent synthetic mechanism: observed features + shared local path structure.
    vulns = pd.DataFrame({'vulnerability_id':[f'V{i:04d}' for i in range(n_vulnerabilities)],
        'severity':severity, 'exploitability':exploitability, 'external_exposure':exposed,
        'asset_criticality':asset_criticality, 'privilege_indicator':privilege,
        'asset_id':[f'A{i%150:03d}' for i in range(n_vulnerabilities)]})
    # Weighted selection creates realistic concentration of attack-path bottlenecks.
    weights = (0.1 + exploitability + 0.5*exposed + 0.25*asset_criticality)
    weights /= weights.sum()
    path_rows=[]; membership=[]
    for p in range(n_paths):
        critical_idx = int(p % n_reachable)
        length = int(rng.integers(1, 4))
        chosen = rng.choice(n_vulnerabilities, size=length, replace=False, p=weights)
        pid=f'P{p:05d}'
        path_rows.append({'path_id':pid, 'entry_id':f'E{p%12:02d}',
                          'critical_asset_id':f'C{critical_idx:03d}', 'vulnerability_count':length})
        for order, v in enumerate(chosen):
            membership.append({'path_id':pid,'position':order+1,'vulnerability_id':f'V{v:04d}'})
    paths=pd.DataFrame(path_rows); members=pd.DataFrame(membership)
    # Labels are simulated exploitation outcomes, NOT measured cyber incidents.
    freq = members.vulnerability_id.value_counts().reindex(vulns.vulnerability_id, fill_value=0).to_numpy()
    neighbor_signal = np.log1p(freq) / np.log1p(max(freq.max(),1))
    logit = -3.0 + 0.22*severity + 2.0*exploitability + 0.75*exposed + 0.55*privilege + 0.65*neighbor_signal
    probability = 1/(1+np.exp(-logit))
    vulns['synthetic_exploitation_label'] = rng.binomial(1, probability)
    return vulns, paths, members

if __name__=='__main__':
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    v,p,m=generate_dataset()
    for name,df in [('vulnerabilities.csv',v),('attack_paths.csv',p),('path_membership.csv',m)]:
        df.to_csv(root/'data'/name,index=False)
