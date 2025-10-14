
import numpy as np
import pandas as pd
import scipy.stats as stats
import igraph as ig
import scanpy as sc

from scipy import sparse

import os
from umap.umap_ import fuzzy_simplicial_set
from scipy.sparse import coo_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score as sil
from scib_metrics.benchmark import Benchmarker
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import precision_recall_curve,roc_curve,auc, average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from scConCRL.utils import get_denoised_adata

try:
    import leidenalg
except ImportError:
    raise ImportError(
        'Please install the leiden algorithm: `conda install -c conda-forge leidenalg` or `pip3 install leidenalg`.'
    )

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#
# related metrics
#
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

## 1. Gene regulatory network inference
def eval_EP_info(ref_path, pred_path):
    output = pd.read_csv(pred_path, sep=',', header=0, index_col=0)
    output['EdgeWeight'] = abs(output['EdgeWeight'])

    output['Gene1'] = output['Gene1'].astype(str)
    output['Gene2'] = output['Gene2'].astype(str)    
    output = output.sort_values('EdgeWeight',ascending=False)
    
    label = pd.read_csv(ref_path, sep=',', index_col=None, header=0)
    label['Gene1'] = label['Gene1'].astype(str)
    label['Gene2'] = label['Gene2'].astype(str)

    TFs = set(label['Gene1']) 
    Genes = set(label['Gene1']) | set(label['Gene2'])
    output = output[output['Gene1'].apply(lambda x: x in TFs)]
    output = output[output['Gene2'].apply(lambda x: x in Genes)]

    label_set = set(label['Gene1']+'|'+label['Gene2'])
    output= output.iloc[:len(label_set)]  # top K predicted edges
    print("k", len(label_set))
    print("possible edge", len(TFs)*len(Genes)-len(TFs))
    EPratio= len(set(output['Gene1']+ '|' +output['Gene2']) & label_set) / (len(label_set)**2/(len(TFs)*len(Genes)-len(TFs)))
    EP = len(set(output['Gene1']+ '|' +output['Gene2']) & label_set) / len(label_set) 
    return EPratio, EP

def eval_AUPR_info(ref_path, pred_path):
    output = pd.read_csv(pred_path, sep=',', header=0, index_col=0)
    output['EdgeWeight'] = abs(output['EdgeWeight'])    
    output['Gene1'] = output['Gene1'].astype(str)
    output['Gene2'] = output['Gene2'].astype(str)
    output = output.sort_values('EdgeWeight',ascending=False)
    
    label = pd.read_csv(ref_path, sep=',', index_col=None, header=0)
    label['Gene1'] = label['Gene1'].astype(str)
    label['Gene2'] = label['Gene2'].astype(str)

    TFs = set(label['Gene1'])
    Genes = set(label['Gene1'])| set(label['Gene2']) - set("AVG")
    # print(len(Genes))
    output = output[output['Gene1'].apply(lambda x: x in TFs)]
    output = output[output['Gene2'].apply(lambda x: x in Genes)]
    label_set = set(label['Gene1']+label['Gene2'])
    # print(len(label_set))

    preds, labels, randoms = [], [], []
    res_d = {}
    l = []
    p= []
    for item in (output.to_dict('records')):
            res_d[item['Gene1']+item['Gene2']] = item['EdgeWeight']
    for item in (set(label['Gene1'])):
            for item2 in  set(label['Gene1'])| set(label['Gene2']):
                if item == item2: # TF * TG - TF
                    continue
                if item+item2 in label_set:
                    l.append(1) 
                else:
                    l.append(0)  
                if item+ item2 in res_d:
                    p.append(res_d[item+item2])
                else:
                    p.append(-1)
    print("possible edge", len(p))
    AUPRratio = average_precision_score(l,p)/np.mean(l) # 等效于网络密度
    AUPR = average_precision_score(l,p,pos_label=1)
    roc = roc_auc_score(l,p)
    return AUPRratio, AUPR, roc

def extractEdgesFromMatrix(m, geneNames):
    geneNames = np.array(geneNames)
    import copy
    mat = copy.deepcopy(m)
    num_nodes = mat.shape[0]
    mat_indicator_all = np.zeros([num_nodes, num_nodes])
    mat_indicator_all[abs(mat) > 0] = 1
    idx_rec, idx_send = np.where(mat_indicator_all)
    edges_df = pd.DataFrame(
        {'Gene1': geneNames[idx_send], 'Gene2': geneNames[idx_rec], 'EdgeWeight': (mat[idx_rec, idx_send])})
    edges_df = edges_df.sort_values('EdgeWeight', ascending=False)
    return edges_df


## 2. Correlation
## 2. Correlation
def eval_corr(clean_data, noisy_data, denoised_data, mask=False):
    if mask: # eval the imputation recovery capability
        corr_zeros = []
        for c in range(noisy_data.shape[0]):
            mask_data = noisy_data[c] == 0
            # print(clean_data[c,mask_data].shape)
            sub_clean_data = clean_data[c,mask_data] / np.linalg.norm(clean_data[c,mask_data])
            sub_denoise_data = denoised_data[c,mask_data] / np.linalg.norm(denoised_data[c,mask_data])
            corr_zeros.append(spearmanr(sub_clean_data, sub_denoise_data)[0])
        corr_zeros_mean = np.mean(np.array(corr_zeros)[~np.isnan(corr_zeros)])
        corr_zeros_std = np.std(np.array(corr_zeros)[~np.isnan(corr_zeros)])
        # print("clean_denoised_corr(zero): ", corr_zeros_mean, corr_zeros_std)
    else:
        corr_zeros = None
    # eval the gene recovery capability
    clean_noisy_list = [spearmanr(clean_data[c,:], noisy_data[c,:])[0] for c in range(clean_data.shape[0])]
    clean_noisy_corr_mean = np.mean(np.array(clean_noisy_list)[~np.isnan(clean_noisy_list)])
    clean_noisy_corr_std = np.std(np.array(clean_noisy_list)[~np.isnan(clean_noisy_list)])
    clean_denoised_list = [spearmanr(clean_data[c,:], denoised_data[c,:])[0] for c in range(clean_data.shape[0])]
    clean_denoised_corr_mean = np.mean(np.array(clean_denoised_list)[~np.isnan(clean_denoised_list)])
    clean_denoised_corr_std = np.std(np.array(clean_denoised_list)[~np.isnan(clean_denoised_list)])
    # print("clean_noisy_corr: ", clean_noisy_corr_mean, clean_noisy_corr_std)
    # print("clean_denoised_corr: ", clean_denoised_corr_mean, clean_denoised_corr_std)
    
    return clean_noisy_corr_mean, clean_denoised_corr_mean, corr_zeros_mean


## 3. Cell-to-Cell similarity
def compute_silhouette(clean_data, noisy_data, denoised_data, labels):
    sil_clean = sil(clean_data, labels=labels)
    sil_noisy = sil(noisy_data, labels=labels)
    sil_denoised = sil(denoised_data, labels=labels)
    # print("sil_clean: ", sil_clean)
    # print("sil_noisy: ", sil_noisy)
    # print("sil_denoised: ", sil_denoised)
    return sil_clean, sil_noisy, sil_denoised


## 3. batch correction
def eval_bc_metrics(adata, label_key, batch_key, layer_name):
    bm = Benchmarker(
    adata,
    batch_key=batch_key,
    label_key=label_key,
    embedding_obsm_keys=layer_name,
    n_jobs=-1,
    )
    bm.benchmark()
    
    df = bm.get_results(min_max_scale=False)
    print(df)
    return df


## 4. DEG
def get_deg_list(adata, cell_type: str = "1", threshold: float = 0.05):
    sc.tl.rank_genes_groups(adata, cell_type, method="wilcoxon")
    deg = {}
    for i in set(adata.obs[cell_type]):
        gene_list = adata.uns["rank_genes_groups"]["names"][str(i)]
        msk = adata.uns["rank_genes_groups"]["pvals"][str(i)] <= threshold
        deg[str(i)] = list(gene_list[msk])
    return deg

def get_topn_deg_list(adata, cell_type: str = "1", n_genes : int=2000):
    sc.tl.rank_genes_groups(adata, cell_type, n_genes=n_genes, method="wilcoxon")
    deg = {}
    for i in set(adata.obs[cell_type]):
        gene_list = adata.uns["rank_genes_groups"]["names"][str(i)]
        deg[str(i)] = list(gene_list)
    print(len(gene_list))
    return deg




#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#
# Benchmark the discovery of key genes
#
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
def compute_auroc(inf, gt):
    """
    Calculate AUROC score
    """
    inf_abs = np.abs(inf)
    gt_abs = np.abs(gt)
    _, _, _, _, AUPRC, AUROC, _ = _compute_auc(inf_abs, gt_abs)
    return AUROC

def compute_auprc(inf, gt):
    """
    Calculate AUPRC score
    """
    inf_abs = np.abs(inf)
    gt_abs = np.abs(gt)
    _, _, _, _, AUPRC, AUROC, _ = _compute_auc(inf_abs, gt_abs)
    return AUPRC

def _compute_auc(inf, gt):
    
    if np.max(inf) == 0:
        return 0, 0, 0, 0, 0, 0, 0
    else:
        fpr, tpr, thresholds = roc_curve(y_true=gt, y_score=inf, pos_label=1)
        
        if len(set(gt)) == 1:
            prec, recall = np.array([0., 1.]), np.array([1., 0.])
        else:
            prec, recall, thresholds = precision_recall_curve(y_true=gt, probas_pred=inf, pos_label=1)

        return prec, recall, fpr, tpr, auc(recall, prec), auc(fpr, tpr), thresholds    

def compute_earlyprec(inf, gt):
    """\
    Description:
    ------------
        Calculate the early precision ratio. 
        Early precision: the fraction of true positives in the top-k edges. 
        Early precision ratio: the ratio of the early precision between estim and random estim.
        directed: the directed estimation or not
    Parameters:
    ------------
        inf: estimated score
        gt: ground truth score
    """
    # take absolute values
    inf = np.abs(inf)
    gt = np.abs(gt)
    # select k value
    num_true = np.sum(gt!=0)
    num_inf = np.sum(inf!=0)
    k = min(num_inf, num_true)
    # find the kth inf weight
    infTopk = inf[np.argsort(inf)[-k]]
    # find the top k index in predict
    topIdx = np.where(inf >= infTopk)[0]
    # find the true edges in ground truth
    trueIdx = np.where(gt!=0)[0]
    intersectionSet = np.intersect1d(topIdx, trueIdx)
    Eprec = len(intersectionSet)/(len(topIdx)+1e-12)
    # Erec = len(intersectionSet)/(len(trueIdx)+1e-12)
    return Eprec

def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""

    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es['weight'] = weights
    except:
        pass
    if g.vcount() != adjacency.shape[0]:
        print( 'Your adjacency matrix contained redundant nodes.' )
    return g

def _compute_connectivities_umap(
    knn_indices,
    knn_dists,
    n_neighbors,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
):
    """\
    This is from umap.fuzzy_simplicial_set [McInnes18]_.
    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    """



    # place holder since we use precompute matrix
    X = coo_matrix(([], ([], [])), shape=(knn_indices.shape[0], 1))
    connectivities = fuzzy_simplicial_set(
        X,
        n_neighbors,
        None,
        None,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
    )

    if isinstance(connectivities, tuple):
        # In umap-learn 0.4, this returns (result, sigmas, rhos)
        connectivities = connectivities[0]

    return connectivities.tocsr()

def leiden_cluster(
    X = None, 
    knn_indices = None,
    knn_dists = None,
    resolution = 30.0,
    random_state = 0,
    n_iterations: int = -1,
    k_neighs = 30,
    sigma = 1,
    affin = None,
    **partition_kwargs):



    partition_kwargs = dict(partition_kwargs)
    
    if affin is None:
        if (knn_indices is None) or (knn_dists is None):
            # X is needed
            if X is None:
                raise ValueError("`X' and `knn_indices & knn_dists', at least one need to be provided.")

            neighbor = NearestNeighbors(n_neighbors = k_neighs)
            neighbor.fit(X)
            # get test connectivity result 0-1 adj_matrix, mode = 'connectivity' by default
            knn_dists, knn_indices = neighbor.kneighbors(X, n_neighbors = k_neighs, return_distance = True)

        affin = _compute_connectivities_umap(knn_indices = knn_indices, knn_dists = knn_dists, n_neighbors = k_neighs, set_op_mix_ratio=1.0, local_connectivity=1.0)
        affin = affin.todense()
        
    partition_type = leidenalg.RBConfigurationVertexPartition
    g = get_igraph_from_adjacency(affin, directed = True)

    partition_kwargs['n_iterations'] = n_iterations
    partition_kwargs['seed'] = random_state
    partition_kwargs['resolution_parameter'] = resolution

    partition_kwargs['weights'] = np.array(g.es['weight']).astype(np.float64)
    part = leidenalg.find_partition(g, partition_type, **partition_kwargs)
    groups = np.array(part.membership)

    return groups

def _ecdf(x):
    '''no frills empirical cdf used in fdrcorrection
    '''
    nobs = len(x)
    return np.arange(1,nobs+1)/float(nobs)

def fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False):
    '''
    pvalue correction for false discovery rate.

    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests.

    Parameters
    ----------
    pvals : array_like, 1d
        Set of p-values of the individual tests.
    alpha : float, optional
        Family-wise error rate. Defaults to ``0.05``.
    method : {'i', 'indep', 'p', 'poscorr', 'n', 'negcorr'}, optional
        Which method to use for FDR correction.
        ``{'i', 'indep', 'p', 'poscorr'}`` all refer to ``fdr_bh``
        (Benjamini/Hochberg for independent or positively
        correlated tests). ``{'n', 'negcorr'}`` both refer to ``fdr_by``
        (Benjamini/Yekutieli for general or negatively correlated tests).
        Defaults to ``'indep'``.
    is_sorted : bool, optional
        If False (default), the p_values will be sorted, but the corrected
        pvalues are in the original order. If True, then it assumed that the
        pvalues are already sorted in ascending order.

    Returns
    -------
    rejected : ndarray, bool
        True if a hypothesis is rejected, False if not
    pvalue-corrected : ndarray
        pvalues adjusted for multiple hypothesis testing to limit FDR

    Notes
    -----
    If there is prior information on the fraction of true hypothesis, then alpha
    should be set to ``alpha * m/m_0`` where m is the number of tests,
    given by the p-values, and m_0 is an estimate of the true hypothesis.
    (see Benjamini, Krieger and Yekuteli)

    The two-step method of Benjamini, Krieger and Yekutiel that estimates the number
    of false hypotheses will be available (soon).

    Both methods exposed via this function (Benjamini/Hochberg, Benjamini/Yekutieli)
    are also available in the function ``multipletests``, as ``method="fdr_bh"`` and
    ``method="fdr_by"``, respectively.

    See also
    --------
    multipletests

    '''
    pvals = np.asarray(pvals)
    assert pvals.ndim == 1, "pvals must be 1-dimensional, that is of shape (n,)"

    if not is_sorted:
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = np.take(pvals, pvals_sortind)
    else:
        pvals_sorted = pvals  # alias

    if method in ['i', 'indep', 'p', 'poscorr']:
        ecdffactor = _ecdf(pvals_sorted)
    elif method in ['n', 'negcorr']:
        cm = np.sum(1./np.arange(1, len(pvals_sorted)+1))   #corrected this
        ecdffactor = _ecdf(pvals_sorted) / cm
##    elif method in ['n', 'negcorr']:
##        cm = np.sum(np.arange(len(pvals)))
##        ecdffactor = ecdf(pvals_sorted)/cm
    else:
        raise ValueError('only indep and negcorr implemented')
    reject = pvals_sorted <= ecdffactor*alpha
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
        reject[:rejectmax] = True

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    del pvals_corrected_raw
    pvals_corrected[pvals_corrected>1] = 1
    if not is_sorted:
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        del pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[pvals_sortind] = reject
        return reject_, pvals_corrected_
    else:
        return reject, pvals_corrected

def wilcoxon_rank_sum(counts_x, counts_y, fdr = False):
    # wilcoxon rank sum test
    ngenes = counts_x.shape[1]
    assert ngenes == counts_y.shape[1]
    pvals = []
    for gene_i in range(ngenes):
        _, pval = stats.ranksums(x = counts_x[:, gene_i].squeeze(), y = counts_y[:, gene_i].squeeze())
        pvals.append(pval)
    pvals = np.array(pvals)

    # fdr correction
    if fdr:
        _, pvals = fdrcorrection(pvals)
    return pvals

def batch_removal_metrics(data, batch_name, state_name, save_path=None):
    denoised_adata = get_denoised_adata(data)
    sc.pp.pca(denoised_adata, svd_solver="arpack")

    bm = Benchmarker(
        denoised_adata,
        batch_key=batch_name,
        label_key=state_name,
        embedding_obsm_keys=['X_pca'],
        n_jobs=-1,
        )
    bm.benchmark()

    df = bm.get_results(min_max_scale=False)

    print(df)
    
    df = df[['Isolated labels', 'KMeans NMI', 'Silhouette label', 'cLISI', 
            'Silhouette batch', 'iLISI', 'KBET', 'Graph connectivity']]
    new_columns = pd.MultiIndex.from_arrays([df.columns, df.iloc[1]])
    df.columns = new_columns
    df = df.drop(index=df.index[1])
    mean_by_type = df.groupby(axis=1, level=1).mean()
    print(mean_by_type)

    if save_path == None:
        return df
    else: 
        # 保存结果
        res_dir = save_path
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        csv_file = os.path.join(res_dir, 'scib_res.csv')

        # 检查CSV文件是否存在
        if not os.path.exists(csv_file):
            df.to_csv(csv_file)
        else:
            # 如果文件存在，则读取文件内容
            res_data = pd.read_csv(csv_file, index_col=0)
            res_data = pd.concat([res_data, df])
            res_data = res_data.drop_duplicates()     
            res_data.to_csv(csv_file) 


def grn_inference_metrics(adj, gene_names, data_dir, sub_task,
                        net_list=['Non-Spec-network.csv', 'Spec-network.csv', 'STR-network.csv', 
                        'lofgof-network.csv', 'network.csv', 'grn.csv']):
    print("===================GRN inference===================")
    pred_path = os.path.join(data_dir, 'rankedEdge_pred.csv')
    res = extractEdgesFromMatrix(adj, gene_names)
    res.to_csv(pred_path)

    if sub_task == "Spec":
        net_list = ['Spec-network.csv']
    elif sub_task == "lofgof":
        net_list = ['lofgof-network.csv']
    elif sub_task == 'NSpec':
        net_list = ['Non-Spec-network.csv']
    elif sub_task == 'STR':
        net_list = ['STR-network.csv']
        
    metrics = {}
    for net in net_list:
        ref_path = os.path.join(data_dir, net)
        if not os.path.exists(ref_path):
            continue
        EPratio, EP = eval_EP_info(ref_path, pred_path)
        AUPRratio, AUPR, roc = eval_AUPR_info(ref_path, pred_path)
        print(net)
        print("EPratio", np.round(EPratio,4))
        print("EP", np.round(EP,4))
        print("AUPRratio", np.round(AUPRratio,4))
        print("AUPR", np.round(AUPR, 4))
        print("AUROC", np.round(roc, 4))
        metrics[net] = {'EPratio': EPratio, 'EP': EP, 'AUPRratio': AUPRratio, 
                        'AUPR': AUPR, 'AUROC': roc}
    return metrics


def visualize_data(adata, colors,  save_path, save_name, use_rep='X',):
    sc.settings.figdir = save_path

    for i in colors:
        adata.obs[i] = adata.obs[i].astype('category')

    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=colors, wspace=0.4, legend_fontsize=14, 
             save=f'{save_name}.pdf',show=False) 

def visualize_diff_data(data, colors, save_path, use_rep=['X', 'X_denoised', 'X_recon', 'X_latent']):
    print("===================visualization========================")
    sc.settings.figdir = save_path
    for rep in use_rep:
        visualize_data(data, colors, save_path, f'_{rep}_data', use_rep=rep)


def concat_ctrl_case_pred_data(adata):
    raw_train_adata = sc.AnnData(X=adata.X, obs=adata.obs, var=adata.var)
    trans_train_adata = sc.AnnData(X=adata.obsm['X_trans'], obs=adata.obs, var=adata.var)
    trans_train_adata.obs['condition'] = ['2'] * trans_train_adata.shape[0]
    trans_train_adata_1 = trans_train_adata[raw_train_adata.obs['condition'] == 0]
    concat_train_adata = raw_train_adata.concatenate(trans_train_adata_1)     
    return concat_train_adata   


def ood_metrics(adata, 
                type_list, 
                conditions,
                save_path,
                condition_name='condition',
                type_name='cell_type',
                top_n_genes=100):

    adata.obs[condition_name] = adata.obs[condition_name].astype(str).apply(lambda x: conditions[x])

    # 1. 计算差异表达基因
    DEGs = list()
    for cell_type in type_list:
        celldata = adata.copy()[(adata.obs[type_name] == cell_type) & 
                        (adata.obs[condition_name].isin([conditions["0"], conditions["1"]]))]
        sc.tl.rank_genes_groups(celldata, groupby=condition_name, n_genes= top_n_genes, 
                                method="wilcoxon")
        diff_genes = celldata.uns["rank_genes_groups"]["names"][conditions["0"]]
        DEGs = DEGs + diff_genes.tolist()
    DEGs = set(DEGs)

    # 2. 筛选差异表达基因
    DEG_adata = adata[:, list(DEGs)].copy()

    # 3. 计算细胞类型相似度
    r2_matrix = np.zeros((len(type_list), len(type_list)))
    for i, cell_type_i in enumerate(type_list):
        print(str(i) + ":" + cell_type_i )
        for j, cell_type_j in enumerate(type_list):
            #  为了减少结果的误差，计算100次R2，并把100次的均值作为这个细胞类型的R2结果。每次计算随机sample 80%的数据来计算
            r_values = np.zeros((1, 100))
            condition_1 = DEG_adata[(DEG_adata.obs[condition_name] == conditions["1"]) & 
                                    (DEG_adata.obs[type_name] == cell_type_i)]
            condition_2 = DEG_adata[(DEG_adata.obs[condition_name] == conditions["2"]) & 
                                    (DEG_adata.obs[type_name] == cell_type_j)]
            for k in range(100):
                condition_1_idx = np.random.choice(range(0, condition_1.shape[0]), int(0.8 * condition_1.shape[0]))
                condition_2_idx = np.random.choice(range(0, condition_2.shape[0]), int(0.8 * condition_2.shape[0]))
                if sparse.issparse(condition_2.X):
                    x = np.average(condition_1.X.A[condition_1_idx], axis=0)
                    y = np.average(condition_2.X.A[condition_2_idx], axis=0)
                else:
                    x = np.average(condition_1.X[condition_1_idx], axis=0)
                    y = np.average(condition_2.X[condition_2_idx], axis=0)
                m, b, r_value, p_value, std_err = stats.linregress(x, y)
                r_values[0, k] = r_value ** 2
            r2_matrix[i,j] = r_values.mean()
    print(pd.DataFrame(r2_matrix))    
    pd.DataFrame(r2_matrix, index=type_list, columns=type_list).to_csv(os.path.join(save_path, f"r2_matrix.csv"))


def stim_correlation_metrics(adata, data_dir, idx=None):
    clean_data = sc.read_csv(os.path.join(data_dir, "clean_data.csv"), first_column_names=True).T
    sc.pp.normalize_total(clean_data, target_sum=10000)
    sc.pp.log1p(clean_data)

    if idx is not None:
        clean_data = clean_data[idx]

    clean_noisy_corr, clean_denoised_corr, corr_zeros = eval_corr(clean_data.X, adata.X, 
                                                            adata.obsm['X_denoised'], True)
    _, gene_corr, _ = eval_corr(clean_data.X.T, adata.X.T, 
                                adata.obsm['X_denoised'].T, True)
    print(f"clean_noisy_corr: {clean_noisy_corr}")
    print(f"clean_denoised_corr: {clean_denoised_corr}")
    print(f"gene_corr: {gene_corr}")
    print(f"corr_zeros: {corr_zeros}")

    sil_clean, sil_noisy, sil_denoised = compute_silhouette(
                                            clean_data.X, adata.X, 
                                            adata.obsm['X_denoised'], 
                                            adata.obs['cellstate'])
    print(f"sil_denoised: {sil_denoised}")

    return {'clean_noisy_corr': clean_noisy_corr,  # baseline
            'clean_denoised_corr': clean_denoised_corr,  # denoised cell corr
            'gene_corr': gene_corr,  # denoised gene corr
            'corr_zeros': corr_zeros, # impute corr
            'sil_denoised': sil_denoised # denoised silhouette
            }


def transfer_metrics(adata, inf, save_path=None, seed=None):
    inf_abs = np.abs(inf)
    inf_df = pd.DataFrame(inf_abs, index=adata.var_names, columns=['infWeight'])

    meta_cells = adata.obs
    counts = adata.X
    counts = np.log1p(counts/(np.sum(counts, axis = 1, keepdims = True) + 1e-6) * 100)
    counts_ctrl = counts[(adata.obs["condition"] == 0),:]
    counts_stim = counts[(adata.obs["condition"] == 1),:]

    pvals = wilcoxon_rank_sum(counts_ctrl, counts_stim, fdr = True)
    assert pvals.shape[0] == 2000
    assert np.min(pvals) >= 0
    # scale to 0-1
    score_wilcoxon = 1 - pvals/np.max(pvals)
    score_wilcoxon  = score_wilcoxon.squeeze()

    wilcoxon_df = pd.DataFrame(score_wilcoxon)
    wilcoxon_df.index = adata.var_names
    wilcoxon_df.columns = ["infWeight"]


    data = pd.concat([wilcoxon_df, inf_df], axis=1)
    data.columns = ['wilcoxon', 'DGDM']
    print(data.head())
    
    if save_path is not None:
        data.to_csv(os.path.join(save_path, f"transfer_metrics_{seed}.csv"))
    from scipy.stats import pearsonr

    inf = np.array(data['DGDM'])
    gt = np.array(data['wilcoxon'])
    # gt[gt < 1] = 0
    gt[pvals > 0.05] = 0
    gt[pvals <= 0.05] = 1

    AUPRC_value = compute_auprc(inf, gt)
    # AUPRCre = bmk.compute_auprc(inf, gt)/(ndiff_genes/ngenes)
    AUROC = compute_auroc(inf, gt)
    Eprec = compute_earlyprec(inf, gt)
    Pearson = pearsonr(inf, gt)[0]

    common_gene = {}
    for j in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
        wilcoxon_df = wilcoxon_df.sort_values(['infWeight'], ascending=False)
        inf_df = inf_df.sort_values(['infWeight'], ascending=False)
        common_gene[j] = len(wilcoxon_df.index[:j] & inf_df.index[:j])
    
    auprc_dict = pd.DataFrame(columns = ["AUPRC", "AUROC", "Eprec", "Pearson", "common_100", "common_200",
                                     "common_300", "common_400", "common_500"])
    auprc_dict = pd.concat([auprc_dict,
                            pd.DataFrame.from_dict({
                                    "AUPRC": [AUPRC_value], 
                                    "AUROC": [AUROC],
                                    "Eprec": [Eprec],
                                    "Pearson": [Pearson],
                                    "common_100": [common_gene[100]],
                                    "common_200": [common_gene[200]],
                                    "common_300": [common_gene[300]],
                                    "common_400": [common_gene[400]],
                                    "common_500": [common_gene[500]],
                                    })], axis = 0, ignore_index = True)

    print(auprc_dict.iloc[:, :5])
    print(auprc_dict.iloc[:, -5:])
    return auprc_dict, data



def net_extract(netFile, gene_index):
    netDF = pd.read_csv(netFile)
    # print(netDF.head())
    netDF = netDF[(netDF.Gene1.isin(gene_index)) & (netDF.Gene2.isin(gene_index))]
    # Remove self-loops.
    netDF = netDF[netDF.Gene1 != netDF.Gene2]
    netDF.drop_duplicates(keep = 'first', inplace=True)
    
    allNodes = set(netDF.Gene1.unique()).union(set(netDF.Gene2.unique()))
    nTFs = len(set(netDF.Gene1.unique()))
    nGenesinNet = len(allNodes)
    print("\n#TFs: %d, #Genes: %d, #Edges: %d, Density: %.3f" % (nTFs,nGenesinNet,netDF.shape[0],netDF.shape[0]/((nTFs*nGenesinNet)-nTFs)))
    
    return netDF

