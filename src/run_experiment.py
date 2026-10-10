"""Run end-to-end synthetic benchmark and save all artifacts."""
from pathlib import Path
import json,platform,sys
import numpy as np
import pandas as pd
import sklearn,networkx,torch,xgboost
from generate_dataset import generate_dataset
from construct_graph import construct_graph,vulnerability_adjacency
from train_models import fit_models
from simulate_remediation import simulate
from evaluate_results import validate

ROOT=Path(__file__).resolve().parents[1]
def run(seed=42):
    (ROOT/'data').mkdir(exist_ok=True);(ROOT/'results').mkdir(exist_ok=True)
    vulns,paths,membership=generate_dataset(seed=seed)
    graph=construct_graph(vulns,paths,membership)
    adjacency=vulnerability_adjacency(vulns,membership)
    pred,metric_rows,splits=fit_models(vulns,adjacency,seed=seed)
    results,selections=simulate(vulns,paths,membership,pred)
    metrics=pd.DataFrame(metric_rows);remediation=pd.DataFrame(results)
    validate(vulns,paths,membership,metrics,remediation)
    for name,frame in [('vulnerabilities',vulns),('attack_paths',paths),('path_membership',membership)]:
        frame.to_csv(ROOT/'data'/f'{name}.csv',index=False)
    metrics.to_csv(ROOT/'results'/'predictive_metrics.csv',index=False)
    remediation.to_csv(ROOT/'results'/'remediation_outcomes.csv',index=False)
    pd.DataFrame(selections).to_csv(ROOT/'results'/'remediation_selections.csv',index=False)
    pd.DataFrame({'vulnerability_id':vulns.vulnerability_id,**pred}).to_csv(ROOT/'results'/'model_scores.csv',index=False)
    pd.DataFrame([{'vulnerability_id':vulns.vulnerability_id.iloc[i], 'split':s} for s,inds in splits.items() for i in inds]).to_csv(ROOT/'results'/'data_splits.csv',index=False)
    metadata={'benchmark_type':'synthetic_simulation_not_real_enterprise_data','seed':seed,
        'initial_paths':len(paths),'critical_assets':100,'initially_reachable_critical_assets':80,
        'vulnerability_count':len(vulns),'graph_nodes':graph.number_of_nodes(),'graph_edges':graph.number_of_edges(),
        'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
        'sklearn':sklearn.__version__,'networkx':networkx.__version__,'torch':torch.__version__,
        'xgboost':xgboost.__version__}
    (ROOT/'results'/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('Synthetic benchmark completed; all validation checks passed.')
    print(metrics.round(3).to_string(index=False))
    print(remediation[remediation.budget==30].round(2).to_string(index=False))
if __name__=='__main__':run()
