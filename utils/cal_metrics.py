from sklearn.metrics import normalized_mutual_info_score, mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score
from metric import jaccard, Dice, F_measure
from sklearn.metrics import silhouette_score
import scanpy as sc

def read_list_from_file(path):
    list = []
    with open(path, 'r') as f:
        for line in f:
            num = int(line.strip())
            list.append(num)

    return list

target = "./datasets/GT_labels.txt"
predict = "./datasets/cluster_label.h5ad"
adata = sc.read_h5ad(predict)
label = adata.obs['fmavae']


GT_list = read_list_from_file(target)
Our_list = label.tolist()
# print(Our_list)

Our_Jaccard = jaccard(Our_list, GT_list)
print(f"our         jaccard: {Our_Jaccard:.6f}")

Our_F_measure = F_measure(Our_list, GT_list)
print(f"our         F_measure: {Our_F_measure:.6f}")

Our_mutual_info = mutual_info_score(GT_list, Our_list)
print(f"our         Mutual Information: {Our_mutual_info:.6f}")

Our_nmi = normalized_mutual_info_score(GT_list, Our_list)
print(f"Our         (NMI): {Our_nmi:.6f}")

Our_ami = adjusted_mutual_info_score(GT_list, Our_list)
print(f"Our         (AMI): {Our_ami:.6f}")

Our_V = v_measure_score(GT_list, Our_list)
print(f"Our         V-measure: {Our_V:.6f}")

Our_homogeneity = homogeneity_score(GT_list, Our_list)

Our_completeness = completeness_score(GT_list, Our_list)
print(f"Our         Homogeneity: {Our_homogeneity:.6f} Completeness: {Our_completeness:.6f}")

Our_ari = adjusted_rand_score(GT_list, Our_list)
print(f"Our         (ARI): {Our_ari:.6f}")

Our_fmi = fowlkes_mallows_score(GT_list, Our_list)
print(f"Our         (FMI): {Our_fmi:.6f}")