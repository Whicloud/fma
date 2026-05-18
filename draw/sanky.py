import scanpy as sc
import pandas as pd
import plotly.graph_objects as go


def plot_sankey_correct(adata, gt_col, ann_col):
    # 1. 统计频数
    df_counts = adata.obs.groupby([gt_col, ann_col]).size().reset_index(name='value')
    
    # 2. 提取唯一的标签并分左右
    gt_labels = sorted(df_counts[gt_col].unique().tolist())
    ann_labels = sorted(df_counts[ann_col].unique().tolist())
    
    gt_nodes  = [f"Ground Truth_{x}" for x in gt_labels]
    ann_nodes = [f"fma_{x}" for x in ann_labels]
    all_nodes = gt_nodes + ann_nodes

    node2label = {}


    dominant_gt = (
        df_counts
        .sort_values('value', ascending=False)
        .drop_duplicates(subset=[ann_col])
        .set_index(ann_col)[gt_col]
    )

    # 左边：GT，本身就是 cell type
    for x in gt_labels:
        node2label[f"Ground Truth_{x}"] = str(x)

    # 右边：PRAGA，用 dominant_gt 映射到 GT
    for x in ann_labels:
        node2label[f"fma_{x}"] = str(dominant_gt[x])
    # 汇总所有节点名
    # all_nodes = gt_labels + ann_labels
    
    # 3. 【核心】强制分配坐标
    # 前一半节点（GT）放左边 x=0.01，后一半（Ann）放右边 x=0.99
    # y 坐标设置为 None 让它在每一列内自动排布
    x_coords = [0.01] * len(gt_labels) + [0.99] * len(ann_labels)
    
    # 4. 建立索引映射
    # 注意：Ann 节点的索引要加上 gt_labels 的长度，确保左右节点完全分开
    source = df_counts[gt_col].apply(lambda x: gt_labels.index(x)).tolist()
    target = df_counts[ann_col].apply(lambda x: ann_labels.index(x) + len(gt_labels)).tolist()
    value = df_counts['value'].tolist()


    # 对每个 PRAGA，找到 value 最大的 GT





    import plotly.colors as pc

    CUSTOM_COLORS = {
        "1": "rgb(30, 118, 179)",   # 蓝
        "2": "rgb(255, 126, 13)",   # 橙
        "3": "rgb(38, 157, 103)",    # 绿
        "4": "rgb(213, 38, 39)",    # 红
        "5": "rgb(169, 63, 251)",  # 紫
        "6": "rgb(139, 86, 74)",    # 棕
        "7": "rgb(226, 118, 193)",  # 粉
        "8": "rgb(180, 188, 96)",  # 灰
        "9": "rgb(22, 189, 206)",  # 灰
        "10": "rgb(173, 198, 231)",  # 灰
        "11": "rgb(255, 186, 119)",  # 灰
        "12": "rgb(151, 222, 137)",  # 灰
        "13": "rgb(255, 151, 149)",  # 灰
        "14": "rgb(196, 175, 212)"  # 灰
    }

    CUSTOM_COLORS_10X = {
        "1": "rgb(30, 118, 179)",   # 蓝
        "2": "rgb(255, 126, 13)",   # 橙
        "3": "rgb(43, 159, 43)",    # 绿
        "4": "rgb(213, 38, 39)",    # 红
        "5": "rgb(147, 102, 188)",  # 紫
        "6": "rgb(139, 86, 74)",    # 棕
        "7": "rgb(226, 118, 193)",  # 粉
        "8": "rgb(148, 148, 148)",  # 灰
        "9": "rgb(188, 188, 33)",  # 灰
        "10": "rgb(22, 189, 206)"  # 灰
    }



    palette = pc.qualitative.Plotly
    labels = sorted(set(node2label.values()))
    label2color = {
        l: palette[i % len(palette)]
        for i, l in enumerate(labels)
    }

    node_colors = [
        CUSTOM_COLORS[node2label[n]]
        for n in all_nodes
    ]

    # colors = pc.qualitative.Plotly + pc.qualitative.Safe
    # node_colors = [colors[i % len(colors)] for i in range(len(all_nodes))]
    
    # 5. 绘图
    fig = go.Figure(data=[go.Sankey(
        arrangement = "fixed", # 固定位置，不允许乱跑
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = all_nodes,
            x = x_coords,  # 强制横向位置
            # y = [...],   # 如果对高度有特殊要求也可以指定
            color = node_colors
        ),
        link = dict(
            source = source,
            target = target,
            value = value,
            color = [node_colors[s].replace('rgb', 'rgba').replace(')', ', 0.4)') for s in source]
        )
    )])

    fig.update_layout(
        # title_text="",
        font=dict(
            family="Times New Roman",
            size=12,
            color="black"
        ),
        width=400,
        height=450,
        margin=dict(l=70, r=70, t=40, b=30)
    )
    fig.update_traces(
        node=dict(
            thickness=10,
            line = dict(color="#888",width=0.5)
        )
    )

    
    return fig




def read_list_from_file(path):
    list = []
    with open(path, 'r') as f:
        for line in f:
            num = int(line.strip())
            list.append(num)

    return list

labels = read_list_from_file('./datasets/GT_labels.txt')
labels = [x + 1 for x in labels]


adata = sc.read_h5ad('./datasets/cluster_label.h5ad')

adata.obs['cell_type'] = labels
adata.obs['cell_type'] = adata.obs['cell_type'].astype('int')
adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')

fig = plot_sankey_correct(adata, gt_col='cell_type', ann_col='fmavae')
fig.write_image(
    "sankey_hln_fma.pdf",
    width=500,
    height=600
)
fig.show()