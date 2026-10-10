"""Greedy sequential remediation, evaluated against fixed synthetic paths."""
import numpy as np

BUDGETS=(10,20,30,50)

def simulate(vulns,paths,membership,predictions,budgets=BUDGETS):
    ids=vulns.vulnerability_id.tolist();idx={v:i for i,v in enumerate(ids)}
    path_ids=paths.path_id.tolist();path_index={p:i for i,p in enumerate(path_ids)}
    path_sets=[set() for _ in path_ids]
    for r in membership.itertuples(index=False):path_sets[path_index[r.path_id]].add(idx[r.vulnerability_id])
    by_v=[set() for _ in ids]
    for p,vs in enumerate(path_sets):
        for v in vs:by_v[v].add(p)
    initial_assets=set(paths.critical_asset_id)
    rows=[];selections=[]
    for model,probs in {**predictions,'CVSS':vulns.severity.to_numpy()/10}.items():
        # Greedy selection recomputes uncovered paths; graph exposure is shared across methods.
        remaining=set(range(len(path_ids)));chosen=set()
        for step in range(1,max(budgets)+1):
            candidates=[v for v in range(len(ids)) if v not in chosen]
            # Vulnerability prioritization: predicted exploitation risk x marginal path coverage.
            score=lambda v:(float(probs[v])*len(by_v[v]&remaining),-v)
            best=max(candidates,key=score)
            chosen.add(best);eliminated=by_v[best]&remaining;remaining-=eliminated
            selections.append({'model':model,'remediation_step':step,'vulnerability_id':ids[best],
                               'marginal_paths_eliminated':len(eliminated),'remaining_paths':len(remaining)})
            if step in budgets:
                reachable=set(paths.iloc[sorted(remaining)].critical_asset_id)
                rows.append({'model':model,'budget':step,'initial_paths':len(paths),
                    'eliminated_paths':len(paths)-len(remaining),'remaining_paths':len(remaining),
                    'path_elimination_pct':100*(len(paths)-len(remaining))/len(paths),
                    'residual_exposure_pct':100*len(remaining)/len(paths),
                    'initial_reachable_critical_assets':len(initial_assets),
                    'remaining_reachable_critical_assets':len(reachable),
                    'critical_reachability_reduction_pct':100*(len(initial_assets)-len(reachable))/len(initial_assets),
                    'paths_eliminated_per_remediation':(len(paths)-len(remaining))/step})
    return rows,selections
