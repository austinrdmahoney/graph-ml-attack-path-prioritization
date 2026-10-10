"""Independent deterministic arithmetic and invariants."""
def validate(vulns,paths,membership,metrics,remediation):
    assert len(paths)==1250 and len(set(paths.critical_asset_id))==80
    assert len(set(vulns.vulnerability_id))==len(vulns)
    assert membership.groupby('path_id').size().between(1,3).all()
    assert set(metrics.model)=={'GNN','XGBoost','Random Forest'}
    for row in remediation.itertuples(index=False):
        assert row.eliminated_paths+row.remaining_paths==row.initial_paths
        assert abs(row.path_elimination_pct+row.residual_exposure_pct-100)<1e-8
        assert 0<=row.remaining_reachable_critical_assets<=row.initial_reachable_critical_assets
    for _,group in remediation.groupby('model'):
        assert group.sort_values('budget').eliminated_paths.is_monotonic_increasing
    return True
