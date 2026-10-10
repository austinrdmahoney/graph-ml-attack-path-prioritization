"""Directed heterogeneous graph and vulnerability co-occurrence projection."""
import networkx as nx
import numpy as np


def construct_graph(vulnerabilities, paths, membership):
    g=nx.DiGraph()
    for row in vulnerabilities.itertuples(index=False):
        g.add_node(row.vulnerability_id,kind='vulnerability')
        g.add_node(row.asset_id,kind='asset')
        g.add_edge(row.asset_id,row.vulnerability_id,relation='contains_vulnerability')
    for row in paths.itertuples(index=False):
        g.add_node(row.entry_id,kind='entry')
        g.add_node(row.critical_asset_id,kind='critical_asset')
    grouped=membership.sort_values('position').groupby('path_id')['vulnerability_id'].apply(list)
    path_lookup=paths.set_index('path_id')
    for pid, ids in grouped.items():
        entry=path_lookup.loc[pid,'entry_id']; target=path_lookup.loc[pid,'critical_asset_id']
        previous=entry
        for i,vid in enumerate(ids):
            # Shared edges encode possible transitions, while membership retains exact routes.
            g.add_edge(previous,vid,relation='exploit_transition')
            previous=vid
        g.add_edge(previous,target,relation='critical_reachability')
    # Identity and privilege nodes are explicit synthetic graph context.
    for i in range(20):
        ident=f'I{i:02d}'; priv=f'R{i%8:02d}'
        g.add_node(ident,kind='identity');g.add_node(priv,kind='privilege')
        g.add_edge(ident,priv,relation='has_privilege')
        g.add_edge(priv,f'A{i%150:03d}',relation='grants_access')
    return g


def vulnerability_adjacency(vulnerabilities, membership):
    """Sparse undirected co-occurrence matrix; no labels or target outcomes."""
    from scipy.sparse import coo_matrix, eye, diags
    ids=vulnerabilities.vulnerability_id.tolist(); index={v:i for i,v in enumerate(ids)}
    rows=[];cols=[]
    for _,group in membership.groupby('path_id'):
        vals=[index[v] for v in group.vulnerability_id]
        for i in vals:
            for j in vals:
                if i!=j:rows.append(i);cols.append(j)
    n=len(ids)
    adj=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n)).tocsr()
    adj.data[:]=1
    adj=adj+eye(n,format='csr')
    degree=np.asarray(adj.sum(axis=1)).ravel()
    d=diags(1/np.sqrt(np.maximum(degree,1)))
    return (d@adj@d).tocsr()
