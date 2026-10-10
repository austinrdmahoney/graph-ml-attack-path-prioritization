# Executable synthetic attack-path prioritization benchmark

**Research integrity:** This package generates and analyzes **synthetic simulated data**, not real-world vulnerability records. Results are computed by the supplied Python implementation, **not** hard-coded to match the manuscript's illustrative figures. They do not independently validate claims about enterprise deployments or superiority in real environments.

## How to run

From the GitHub website, upload the extracted `src/`, `requirements.txt`, `data/`, and `results/` files into the research repository (GitHub's web uploader supports folders when dragging an extracted folder, depending on browser). To rerun without a local terminal, use a Python notebook environment such as Google Colab, or configure a GitHub Actions workflow. In a Python environment with dependencies installed, run `python src/run_experiment.py` from the package root. No Git CLI is required.

## Experimental design

- 300 synthetic vulnerability records, 1,250 explicitly enumerated attack paths, 100 designated critical assets, 80 initially reachable critical assets, 150 asset identifiers, 12 entry points.
- Directed heterogeneous graph includes entry, asset, vulnerability, identity, privilege, and critical-asset nodes. The `path_membership.csv` file, rather than all possible graph traversals, is the **authoritative set of simulated attack paths**; joining graph edges can otherwise create extra unintended routes.
- Seed 42; one generated benchmark and one split, 60% train / 15% validation / 25% held-out test. Labels are Bernoulli draws from a documented synthetic exploitation mechanism; they are not observed exploits.
- GNN: 2-layer graph convolution over vulnerability co-occurrence adjacency, with transductive visibility of the full unlabeled graph; tabular Random Forest and XGBoost use the same five numeric input features but do not receive graph edges. This **is not a perfectly feature-equivalent comparison**.
- Classification metrics are measured on held-out synthetic vulnerability labels. Remediation uses a greedy policy maximizing predicted exploitation probability times *currently uncovered* path coverage; CVSS comparator uses severity/10 in the same policy. This is a severity-only proxy, **not a complete CVSS v3.1/v4 implementation**.
- Remediation removes every explicitly enumerated path containing the selected vulnerability. Critical-asset reachability is defined by whether any enumerated path to the asset remains. Budgets: 10, 20, 30, 50.
- No statistical significance or uncertainty claim: one seed and one split. No external validation. A rigorous paper should add repeated seeds, confidence intervals, ablations, data provenance, and more realistic adversary and remediation-cost modeling.

## Files

- `src/generate_dataset.py`: synthetic records, labels, path membership
- `src/construct_graph.py`: directed heterogeneous graph and vulnerability adjacency
- `src/train_models.py`: GNN, XGBoost, Random Forest and held-out classification metrics
- `src/simulate_remediation.py`: path-disruption policy and budget comparison
- `src/evaluate_results.py`: arithmetic and monotonicity assertions
- `src/run_experiment.py`: complete reproducible pipeline
- `data/*.csv`: generated benchmark inputs
- `results/*.csv`: **actually executed synthetic benchmark outputs** and split assignments
- `results/run_metadata.json`: execution environment and parameters

**Do not replace manuscript illustrative results with these numbers without revising the study's methods, limitations, results, and claims.**
