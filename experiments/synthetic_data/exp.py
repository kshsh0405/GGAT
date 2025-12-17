import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.utils import add_remaining_self_loops, to_undirected, coalesce, softmax
from torch_geometric.data import Data
from typing import Optional, Union, List
from utils.graph_signature_index import *
from models.model import *
from synthetic_data.data import *
import warnings
warnings.filterwarnings('ignore')

# ============== Utility Functions ==============
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(num_params):
    if num_params >= 1e6:
        return f"{num_params/1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params/1e3:.1f}K"
    else:
        return str(num_params)


# ============== Training ==============
def train_and_evaluate(model, data, config, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0

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

            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config['patience']:
                break

    return best_test_acc


# ============== Main Experiment ==============
def run_homophily_experiment(verbose=True):
    set_seed(42)

    # Experiment settings
    conv_types = ['ScaledDot', 'GATConv', 'GATv2Conv', 'GGAT']
    homophily_levels = np.linspace(0.0, 1, 21)
    num_runs = 5

    # Synthetic data parameters
    data_config = {
        'n': 500,
        'm': 1000,
        'num_classes': 5,
        'm1': 20,
        'm2': 25,
        'num_edges': 1000,
        'feature_noise_std': 0.4
    }

    # Model config
    model_config = {
        'nhid': 256,
        'nlayers': 2,
        'heads': 4,
        'drop_in': 0.6,
        'drop': 0.6,
        'lr': 0.002,
        'weight_decay': 5e-4,
        'epochs': 1000,
        'patience': 100,
        'init_type': 'uniform'
    }

    # Results storage
    results = {ct: {h: [] for h in homophily_levels} for ct in conv_types}

    print("="*80)
    print("HOMOPHILY EXPERIMENT: Synthetic Dataset")
    print(f"Data: n={data_config['n']}, m={data_config['m']}, classes={data_config['num_classes']}")
    print(f"Model: nhid={model_config['nhid']}, nlayers={model_config['nlayers']}, heads={model_config['heads']}")
    print(f"GGAT: init_type={model_config['init_type']}")
    print("="*80)

    for h_idx, h in enumerate(homophily_levels):
        print(f"\n--- Homophily = {h:.2f} ---")

        # 각 homophily level에서 데이터셋은 한 번만 생성 (seed 고정)
        data = generate_synthetic_heterophilic_dataset(
            n=data_config['n'],
            m=data_config['m'],
            num_classes=data_config['num_classes'],
            m1=data_config['m1'],
            m2=data_config['m2'],
            num_edges=data_config['num_edges'],
            homophily_ratio=h,
            feature_noise_std=data_config['feature_noise_std'],
            seed=42  # 고정된 seed로 동일한 데이터셋 생성
        )

        actual_homophily = compute_homophily(data)
        
        # Preprocess graph
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        edge_index = coalesce(edge_index, num_nodes=num_nodes)
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
        data.edge_index = edge_index

        

        nfeat = data.x.size(1)
        nclass = data.num_classes

        for run in range(num_runs):
            # 매 run마다 다른 split 생성
            train_mask, val_mask, test_mask = create_random_split(
                data.num_nodes,
                train_ratio=0.6,
                valid_ratio=0.2,
                test_ratio=0.2,
                seed=42 + run
            )
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask

            run_results = {}

            for conv_type in conv_types:
                set_seed(42)

                model = PlainGNN(
                    nfeat, model_config['nhid'], nclass, model_config['nlayers'],
                    conv_type=conv_type, heads=model_config['heads'],
                    drop_in=model_config['drop_in'], drop=model_config['drop'],
                    init_type=model_config['init_type']
                )

                acc = train_and_evaluate(model, data, model_config, verbose=False)
                results[conv_type][h].append(acc)
                run_results[conv_type] = acc

            best_model = max(run_results, key=run_results.get)
            print(f"  Run {run}: h_actual={actual_homophily:.2f} | Best: {best_model} ({run_results[best_model]*100:.1f}%)")

    return results, homophily_levels


def print_results(results, homophily_levels):
    conv_types = ['ScaledDot', 'GATConv', 'GATv2Conv', 'GGAT']

    print("\n" + "="*100)
    print("RESULTS TABLE (Test Accuracy: mean ± std)")
    print("="*100)

    header = f"{'Homophily':<12}"
    for ct in conv_types:
        header += f"{ct:^20}"
    print(header)
    print("-"*100)

    for h in homophily_levels:
        row = f"{h:<12.2f}"
        best_acc = 0
        best_model = ""
        for ct in conv_types:
            accs = results[ct][h]
            mean_acc = np.mean(accs) * 100
            std_acc = np.std(accs) * 100
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_model = ct
            row += f"{mean_acc:.1f}±{std_acc:.1f}".center(20)
        print(row + f"  Best: {best_model}")

    print("="*100)

    # Summary by homophily range
    print("\n" + "="*80)
    print("SUMMARY BY HOMOPHILY RANGE")
    print("="*80)

    ranges = [
        ("Low (0.0-0.3)", [h for h in homophily_levels if h <= 0.3]),
        ("Medium (0.4-0.6)", [h for h in homophily_levels if 0.4 <= h <= 0.6]),
        ("High (0.7-1.0)", [h for h in homophily_levels if h >= 0.7])
    ]

    for range_name, h_list in ranges:
        print(f"\n{range_name}:")
        for ct in conv_types:
            all_accs = [acc for h in h_list for acc in results[ct][h]]
            mean_acc = np.mean(all_accs) * 100
            std_acc = np.std(all_accs) * 100
            print(f"  {ct:15}: {mean_acc:.1f}±{std_acc:.1f}%")


def plot_results(results, homophily_levels):
    conv_types = ['ScaledDot', 'GATConv', 'GATv2Conv', 'GGAT']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    markers = ['o', 's', '^', 'D']

    # Figure 1: Line plot with error bars
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, ct in enumerate(conv_types):
        means = [np.mean(results[ct][h]) * 100 for h in homophily_levels]
        stds = [np.std(results[ct][h]) * 100 for h in homophily_levels]

        ax.errorbar(homophily_levels, means, yerr=stds,
                   label=ct, color=colors[i], marker=markers[i],
                   linewidth=2, markersize=8, capsize=4, capthick=2)

    ax.set_xlabel('Homophily Ratio', fontsize=14)
    ax.set_ylabel('Test Accuracy (%)', fontsize=14)
    ax.set_title('Attention Mechanism Comparison across Homophily Levels', fontsize=16)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(homophily_levels[::2])
    ax.set_xlim(-0.05, 1.05)

    # Highlight regions
    ax.axvspan(-0.05, 0.3, alpha=0.1, color='red', label='Heterophilic')
    ax.axvspan(0.7, 1.05, alpha=0.1, color='blue', label='Homophilic')

    plt.tight_layout()
    plt.savefig('homophily_comparison_line.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Figure 2: Heatmap
    fig, ax = plt.subplots(figsize=(14, 5))

    data_matrix = np.array([[np.mean(results[ct][h]) * 100 for h in homophily_levels] for ct in conv_types])

    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=data_matrix.min()-5, vmax=data_matrix.max()+5)

    ax.set_xticks(np.arange(len(homophily_levels)))
    ax.set_yticks(np.arange(len(conv_types)))
    ax.set_xticklabels([f'{h:.2f}' for h in homophily_levels], rotation=45)
    ax.set_yticklabels(conv_types)
    ax.set_xlabel('Homophily Ratio', fontsize=12)
    ax.set_title('Test Accuracy (%) Heatmap', fontsize=14)

    for i in range(len(conv_types)):
        for j in range(len(homophily_levels)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=8)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va="bottom")

    plt.tight_layout()
    plt.savefig('homophily_comparison_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Figure 3: Bar chart for low vs high homophily
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Low homophily (0.0-0.3)
    ax1 = axes[0]
    low_h = [h for h in homophily_levels if h <= 0.3]
    low_means = [np.mean([acc for h in low_h for acc in results[ct][h]]) * 100 for ct in conv_types]
    low_stds = [np.std([acc for h in low_h for acc in results[ct][h]]) * 100 for ct in conv_types]

    bars1 = ax1.bar(conv_types, low_means, yerr=low_stds, color=colors, capsize=5, alpha=0.8)
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Low Homophily (0.0-0.3)\n(Heterophilic)', fontsize=12)
    ax1.set_ylim(0, 100)
    for bar, mean in zip(bars1, low_means):
        ax1.annotate(f'{mean:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # High homophily (0.7-1.0)
    ax2 = axes[1]
    high_h = [h for h in homophily_levels if h >= 0.7]
    high_means = [np.mean([acc for h in high_h for acc in results[ct][h]]) * 100 for ct in conv_types]
    high_stds = [np.std([acc for h in high_h for acc in results[ct][h]]) * 100 for ct in conv_types]

    bars2 = ax2.bar(conv_types, high_means, yerr=high_stds, color=colors, capsize=5, alpha=0.8)
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('High Homophily (0.7-1.0)\n(Homophilic)', fontsize=12)
    ax2.set_ylim(0, 100)
    for bar, mean in zip(bars2, high_means):
        ax2.annotate(f'{mean:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('homophily_comparison_bar.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Figure 4: Performance difference from baseline (ScaledDot)
    fig, ax = plt.subplots(figsize=(12, 6))

    baseline = 'ScaledDot'
    for i, ct in enumerate(conv_types):
        if ct == baseline:
            continue
        diffs = [np.mean(results[ct][h]) * 100 - np.mean(results[baseline][h]) * 100 for h in homophily_levels]
        ax.plot(homophily_levels, diffs, label=f'{ct} - {baseline}',
               color=colors[i], marker=markers[i], linewidth=2, markersize=8)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Homophily Ratio', fontsize=14)
    ax.set_ylabel('Accuracy Difference (%)', fontsize=14)
    ax.set_title(f'Performance Difference vs {baseline}', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(homophily_levels[::2])

    plt.tight_layout()
    plt.savefig('homophily_comparison_diff.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nPlots saved: homophily_comparison_line.png, homophily_comparison_heatmap.png, "
          "homophily_comparison_bar.png, homophily_comparison_diff.png")


# Run experiment
results, homophily_levels = run_homophily_experiment(verbose=True)
print_results(results, homophily_levels)
plot_results(results, homophily_levels)
