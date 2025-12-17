import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from torch_geometric.utils import add_remaining_self_loops, to_undirected, coalesce, softmax
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB, Actor
from torch_geometric.data import Data
from utils.graph_signature_index import *
from models.model import *
from benchmark_data.data import *
import warnings
warnings.filterwarnings('ignore')

import random

def set_seed(seed):
    """재현성을 위한 seed 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============== Parameter Counter ==============
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(num_params):
    if num_params >= 1e6:
        return f"{num_params/1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params/1e3:.1f}K"
    else:
        return str(num_params)


# ============== Graph Preprocessing ==============
def preprocess_graph(data):
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)

    data.edge_index = edge_index
    return data

# ============== Dataset-specific Hyperparameters ==============
DATASET_CONFIGS = {
    'cornell': {
        'nhid': 128, 'nlayers': 2, 'heads': 4,
        'drop_in': 0.3, 'drop': 0.3,
        'lr': 0.005, 'weight_decay': 5e-4,
        'epochs': 1000, 'patience': 100,
    },
    'texas': {
        'nhid': 128, 'nlayers': 2, 'heads': 4,
        'drop_in': 0.3, 'drop': 0.3,
        'lr': 0.005, 'weight_decay': 5e-4,
        'epochs': 1000, 'patience': 100,
    },
    'chameleon': {
        'nhid': 256, 'nlayers': 2, 'heads': 8,
        'drop_in': 0.5, 'drop': 0.5,
        'lr': 0.005, 'weight_decay': 5e-4,
        'epochs': 1000, 'patience': 200,
    },
    'squirrel': {
        'nhid': 256, 'nlayers': 2, 'heads': 8,
        'drop_in': 0.5, 'drop': 0.5,
        'lr': 0.005, 'weight_decay': 5e-4,
        'epochs': 1000, 'patience': 200,
    },
    'film': {
        'nhid': 128, 'nlayers': 2, 'heads': 4,
        'drop_in': 0.4, 'drop': 0.4,
        'lr': 0.005, 'weight_decay': 5e-4,
        'epochs': 1000, 'patience': 100,
    },
    'cora': {
        'nhid': 128, 'nlayers': 2, 'heads': 4,
        'drop_in': 0.5, 'drop': 0.5,
        'lr': 0.005, 'weight_decay': 5e-4,
        'epochs': 1000, 'patience': 100,
    },
    'citeseer': {
        'nhid': 256, 'nlayers': 2, 'heads': 4,
        'drop_in': 0.5, 'drop': 0.5,
        'lr': 0.005, 'weight_decay': 5e-4,
        'epochs': 1000, 'patience': 100,
    },
    'pubmed': {
        'nhid': 128, 'nlayers': 2, 'heads': 4,
        'drop_in': 0.4, 'drop': 0.4,
        'lr': 0.005, 'weight_decay': 5e-4,
        'epochs': 1000, 'patience': 100,
    }
}

# ============== Parameter Comparison ==============
def print_all_dataset_params():
    conv_types = ['ScaledDot', 'GATConv', 'GATv2Conv', 'GGAT', 'GGAT_FullMetric', 'GGAT_SignatureDecomp']

    all_datasets = ['cornell', 'texas', 'chameleon', 'squirrel', 'film', 'cora', 'citeseer', 'pubmed']

    print("\n" + "="*100)
    print("PARAMETER COUNT BY DATASET (Plain GNN)")
    print("="*100)

    header = f"{'Dataset':<12} {'nfeat':>6} {'nclass':>6} {'nhid':>5}"
    for ct in conv_types:
        header += f"{ct:>14}"
    print(header)
    print("-"*100)

    for ds_name in all_datasets:
        if ds_name in ['cora', 'citeseer', 'pubmed']:
            data = get_data(ds_name)
        else:
            data = get_data(ds_name, split=0)
        preprocess_graph(data)

        nfeat = data.x.size(1)
        nclass = data.y.max().item() + 1
        config = DATASET_CONFIGS[ds_name]

        row = f"{ds_name:<12} {nfeat:>6} {nclass:>6} {config['nhid']:>5}"

        for conv_type in conv_types:
            model = PlainGNN(
                nfeat, config['nhid'], nclass, config['nlayers'],
                conv_type=conv_type, heads=config['heads'],
                drop_in=config['drop_in'], drop=config['drop']
            )
            num_params = count_parameters(model)
            row += f"{format_params(num_params):>14}"

        print(row)

    print("="*100)


# ============== Training ==============
def train_and_evaluate(model, data, config, verbose=False, model_name=""):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0

    log_interval = max(config['epochs'] // 10, 1)

    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.argmax(dim=1)

            train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch % log_interval == 0 or epoch == config['epochs'] - 1):
                print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                      f"Train: {train_acc*100:.1f}% | Val: {val_acc*100:.1f}% | "
                      f"Test: {test_acc*100:.1f}% | Best: {best_test_acc*100:.1f}%")

            if patience_counter >= config['patience']:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

    return best_test_acc


# ============== Main Experiment ==============
def run_experiment(verbose=True):
    set_seed(42)

    conv_types = ['ScaledDot', 'GATConv', 'GATv2Conv', 'GGAT', 'GGAT_FullMetric', 'GGAT_SignatureDecomp']

    datasets_hetero = ['cornell', 'texas', 'chameleon', 'squirrel', 'film']
    datasets_homo = ['cora', 'citeseer', 'pubmed']
    all_datasets = datasets_hetero + datasets_homo

    results = {ds: {ct: [] for ct in conv_types} for ds in all_datasets}
    params_info = {ds: {} for ds in all_datasets}

    print_all_dataset_params()

    # Heterophilic datasets
    for ds_name in datasets_hetero:
        config = DATASET_CONFIGS[ds_name]

        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name.upper()}")
        print(f"Config: nhid={config['nhid']}, nlayers={config['nlayers']}, "
              f"heads={config['heads']}, drop={config['drop']}, lr={config['lr']}")
        print('='*70)

        for split_idx in range(10):
            data = get_data(ds_name, split=split_idx)
            nfeat = data.x.size(1)
            nclass = data.y.max().item() + 1

            if verbose and split_idx == 0:
                print(f"Nodes: {data.num_nodes}, Edges: {data.edge_index.size(1)}, "
                      f"Features: {nfeat}, Classes: {nclass}")

            split_results = {}

            for conv_type in conv_types:
                set_seed(42)

                model = PlainGNN(
                    nfeat, config['nhid'], nclass, config['nlayers'],
                    conv_type=conv_type, heads=config['heads'],
                    drop_in=config['drop_in'], drop=config['drop']
                )

                if split_idx == 0:
                    params_info[ds_name][conv_type] = count_parameters(model)

                if verbose and split_idx == 0:
                    num_params = count_parameters(model)
                    print(f"\n--- {conv_type} ({format_params(num_params)}) (Split {split_idx}) ---")

                acc = train_and_evaluate(
                    model, data, config,
                    verbose=(verbose and split_idx == 0),
                    model_name=conv_type
                )
                results[ds_name][conv_type].append(acc)
                split_results[conv_type] = acc

            best_model = max(split_results, key=split_results.get)
            best_acc = split_results[best_model]

            print(f"Split {split_idx} completed - Best: {best_model} ({best_acc*100:.1f}%)")

    # Homophilic datasets
    for ds_name in datasets_homo:
        config = DATASET_CONFIGS[ds_name]

        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name.upper()}")
        print(f"Config: nhid={config['nhid']}, nlayers={config['nlayers']}, "
              f"heads={config['heads']}, drop={config['drop']}, lr={config['lr']}")
        print('='*70)

        data = get_data(ds_name)
        nfeat = data.x.size(1)
        nclass = data.y.max().item() + 1
        splits = create_random_splits(data, num_splits=10, seed=42)

        if verbose:
            print(f"Nodes: {data.num_nodes}, Edges: {data.edge_index.size(1)}, "
                  f"Features: {nfeat}, Classes: {nclass}")

        for split_idx, (train_mask, val_mask, test_mask) in enumerate(splits):
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask

            split_results = {}

            for conv_type in conv_types:
                set_seed(42)

                model = PlainGNN(
                    nfeat, config['nhid'], nclass, config['nlayers'],
                    conv_type=conv_type, heads=config['heads'],
                    drop_in=config['drop_in'], drop=config['drop']
                )

                if split_idx == 0:
                    params_info[ds_name][conv_type] = count_parameters(model)

                if verbose and split_idx == 0:
                    num_params = count_parameters(model)
                    print(f"\n--- {conv_type} ({format_params(num_params)}) (Split {split_idx}) ---")

                acc = train_and_evaluate(
                    model, data, config,
                    verbose=(verbose and split_idx == 0),
                    model_name=conv_type
                )
                results[ds_name][conv_type].append(acc)
                split_results[conv_type] = acc

            best_model = max(split_results, key=split_results.get)
            best_acc = split_results[best_model]

            print(f"Split {split_idx} completed - Best: {best_model} ({best_acc*100:.1f}%)")

    return results, params_info


def print_results(results):
    conv_types = list(list(results.values())[0].keys())
    
    print("\n" + "="*120)
    print("FINAL RESULTS (Test Accuracy: mean ± std)")
    print("="*120)

    header = f"{'Dataset':<12}"
    for ct in conv_types:
        header += f"{ct:^18}"
    print(header)
    print("-"*120)

    for ds_name in results.keys():
        row = f"{ds_name:<12}"
        best_acc = 0
        best_model = ""
        
        for ct in conv_types:
            accs = results[ds_name][ct]
            mean_acc = np.mean(accs) * 100
            std_acc = np.std(accs) * 100
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_model = ct
            row += f"{mean_acc:.1f}±{std_acc:.1f}".center(18)
        
        print(row + f" Best: {best_model}")

    print("="*120)

# ============== Signature Analysis ==============
def analyze_learned_signatures(model, dataset_name):
    """학습된 모델의 signature 분석"""
    print(f"\n[Signature Analysis: {dataset_name}]")
    
    signatures = model.get_all_signatures()
    for layer_info in signatures:
        print(f"  Layer {layer_info['layer']}:")
        for h, sig in enumerate(layer_info['signatures']):
            print(f"    Head {h}: s = {sig} (pos, zero, neg)")
    
    # SignatureDecomp인 경우 추가 분석
    for conv in model.convs:
        if hasattr(conv, 'get_pos_neg_ratio'):
            ratio_info = conv.get_pos_neg_ratio()
            print(f"  S⁺/S⁻ ratio: {ratio_info['ratio']}")

def plot_results(results, params_info=None):
    """결과 시각화 - 새로운 모델들 포함"""
    
    # 동적으로 conv_types 가져오기
    conv_types = list(list(results.values())[0].keys())
    
    # 색상 (모델 수에 맞게 확장)
    color_palette = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    colors = color_palette[:len(conv_types)]

    datasets_hetero = ['cornell', 'texas', 'chameleon', 'squirrel', 'film']
    datasets_homo = ['cora', 'citeseer', 'pubmed']
    
    # 실제 존재하는 데이터셋만 필터링
    datasets_hetero = [ds for ds in datasets_hetero if ds in results]
    datasets_homo = [ds for ds in datasets_homo if ds in results]
    all_datasets = datasets_hetero + datasets_homo

    # ================================================================
    # Figure 1: 데이터셋별 성능 비교 (Bar chart)
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    width = 0.8 / len(conv_types)  # 동적 width

    # Heterophilic
    if datasets_hetero:
        ax1 = axes[0]
        x = np.arange(len(datasets_hetero))

        for i, ct in enumerate(conv_types):
            means = [np.mean(results[ds][ct]) * 100 for ds in datasets_hetero]
            stds = [np.std(results[ds][ct]) * 100 for ds in datasets_hetero]
            offset = (i - len(conv_types)/2 + 0.5) * width
            ax1.bar(x + offset, means, width, label=ct, color=colors[i], yerr=stds, capsize=2)

        ax1.set_xlabel('Dataset', fontsize=12)
        ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax1.set_title('Heterophilic Datasets', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(datasets_hetero)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 100)

    # Homophilic
    if datasets_homo:
        ax2 = axes[1]
        x = np.arange(len(datasets_homo))

        for i, ct in enumerate(conv_types):
            means = [np.mean(results[ds][ct]) * 100 for ds in datasets_homo]
            stds = [np.std(results[ds][ct]) * 100 for ds in datasets_homo]
            offset = (i - len(conv_types)/2 + 0.5) * width
            ax2.bar(x + offset, means, width, label=ct, color=colors[i], yerr=stds, capsize=2)

        ax2.set_xlabel('Dataset', fontsize=12)
        ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax2.set_title('Homophilic Datasets', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(datasets_homo)
        ax2.legend(loc='lower right', fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig('performance_comparison_bar.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ================================================================
    # Figure 2: 평균 성능 비교 (Grouped bar)
    # ================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(conv_types))
    width = 0.35

    if datasets_hetero:
        hetero_means = [np.mean([np.mean(results[ds][ct]) for ds in datasets_hetero]) * 100 
                        for ct in conv_types]
        bars1 = ax.bar(x - width/2, hetero_means, width, label='Heterophilic', 
                       color='#e74c3c', alpha=0.8)
    
    if datasets_homo:
        homo_means = [np.mean([np.mean(results[ds][ct]) for ds in datasets_homo]) * 100 
                      for ct in conv_types]
        bars2 = ax.bar(x + width/2, homo_means, width, label='Homophilic', 
                       color='#3498db', alpha=0.8)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Average Test Accuracy (%)', fontsize=12)
    ax.set_title('Average Performance: Heterophilic vs Homophilic', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conv_types, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 값 표시
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    if datasets_hetero:
        add_labels(bars1)
    if datasets_homo:
        add_labels(bars2)

    plt.tight_layout()
    plt.savefig('performance_comparison_avg.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ================================================================
    # Figure 3: Heatmap
    # ================================================================
    fig, ax = plt.subplots(figsize=(max(10, len(conv_types)*1.5), max(8, len(all_datasets)*0.8)))

    data_matrix = np.array([[np.mean(results[ds][ct]) * 100 for ct in conv_types] 
                            for ds in all_datasets])

    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=20, vmax=95)

    ax.set_xticks(np.arange(len(conv_types)))
    ax.set_yticks(np.arange(len(all_datasets)))
    ax.set_xticklabels(conv_types, rotation=30, ha='right')
    ax.set_yticklabels(all_datasets)

    # 값 표시
    for i in range(len(all_datasets)):
        for j in range(len(conv_types)):
            val = data_matrix[i, j]
            text_color = 'white' if val < 40 or val > 80 else 'black'
            ax.text(j, i, f'{val:.1f}', ha="center", va="center", 
                   color=text_color, fontsize=9, fontweight='bold')

    # Hetero/Homo 구분선
    if datasets_hetero and datasets_homo:
        ax.axhline(y=len(datasets_hetero) - 0.5, color='white', linewidth=3)

    ax.set_title('Test Accuracy (%) - All Datasets', fontsize=14)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va="bottom")

    plt.tight_layout()
    plt.savefig('performance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ================================================================
    # Figure 4: Radar chart
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(polar=True))

    for ax, (ds_type, ds_list) in zip(axes, [('Heterophilic', datasets_hetero), 
                                              ('Homophilic', datasets_homo)]):
        if not ds_list:
            ax.set_visible(False)
            continue
            
        angles = np.linspace(0, 2 * np.pi, len(ds_list), endpoint=False).tolist()
        angles += angles[:1]

        for i, ct in enumerate(conv_types):
            values = [np.mean(results[ds][ct]) * 100 for ds in ds_list]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=ct, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(ds_list)
        ax.set_title(f'{ds_type} Datasets', size=14, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), fontsize=8)
        ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig('performance_radar.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ================================================================
    # Figure 5: Win Count (NEW)
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    
    win_count = {ct: 0 for ct in conv_types}
    for ds_name in all_datasets:
        best_acc = 0
        best_model = ""
        for ct in conv_types:
            mean_acc = np.mean(results[ds_name][ct])
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_model = ct
        win_count[best_model] += 1

    models = list(win_count.keys())
    wins = list(win_count.values())
    
    bars = ax.barh(models, wins, color=colors)
    ax.set_xlabel('Number of Wins', fontsize=12)
    ax.set_title('Win Count: Best Model per Dataset', fontsize=14)
    ax.set_xlim(0, len(all_datasets) + 1)
    
    for bar, win in zip(bars, wins):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                str(win), va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('win_count.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ================================================================
    # Figure 6: Signature Analysis (NEW - for GGAT variants)
    # ================================================================
    # 이건 학습된 모델이 필요하므로 별도 함수로 분리 권장

    print("\nPlots saved:")
    print("  - performance_comparison_bar.png")
    print("  - performance_comparison_avg.png")
    print("  - performance_heatmap.png")
    print("  - performance_radar.png")
    print("  - win_count.png")


def plot_signature_analysis(model, dataset_name, save_path=None):
    """학습된 모델의 signature 시각화"""
    
    if not hasattr(model, 'get_all_signatures'):
        print(f"Model doesn't support signature analysis")
        return
    
    signatures = model.get_all_signatures()
    if not signatures:
        print("No signature data available")
        return
    
    fig, axes = plt.subplots(1, len(signatures), figsize=(5*len(signatures), 4))
    if len(signatures) == 1:
        axes = [axes]
    
    for ax, layer_info in zip(axes, signatures):
        layer_idx = layer_info['layer']
        sigs = layer_info['signatures']
        
        heads = len(sigs)
        x = np.arange(heads)
        width = 0.25
        
        pos = [s[0] for s in sigs]
        zero = [s[1] for s in sigs]
        neg = [s[2] for s in sigs]
        
        ax.bar(x - width, pos, width, label='Positive', color='#2ecc71')
        ax.bar(x, zero, width, label='Zero', color='#95a5a6')
        ax.bar(x + width, neg, width, label='Negative', color='#e74c3c')
        
        ax.set_xlabel('Head')
        ax.set_ylabel('Count')
        ax.set_title(f'Layer {layer_idx} Eigenvalue Signature')
        ax.set_xticks(x)
        ax.set_xticklabels([f'H{i}' for i in range(heads)])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Learned Metric Signature: {dataset_name}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# Run
results, params_info = run_experiment(verbose=True)
print_results(results)
plot_results(results, params_info)

