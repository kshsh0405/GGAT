import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import add_remaining_self_loops, to_undirected, coalesce, softmax
from torch_geometric.data import Data
from typing import Optional, Union, List
import warnings
warnings.filterwarnings('ignore')

# ============== Synthetic Data Generation ==============
def generate_synthetic_heterophilic_dataset(
    n: int,
    m: int,
    num_classes: int,
    m1: int,
    m2: int,
    num_edges: int,
    homophily_ratio: float = 0.1,
    bernoulli_params: Optional[Union[float, torch.Tensor]] = None,
    class_distribution: Optional[torch.Tensor] = None,
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2,
    test_ratio: float = 0.2,
    add_self_loops: bool = True,
    feature_noise_std: float = 0.0,
    feature_flip_prob: float = 0.0,
    label_noise_ratio: float = 0.0,
    class_feature_similarity: float = 0.0,
    uninformative_feature_ratio: float = 0.0,
    seed: Optional[int] = None
) -> Data:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


    # Generate class-specific coordinate sets with randomization
    # 1. 인덱스 셔플
    perm = torch.randperm(m).tolist()

    # 2. Shared coords의 base (모든 클래스가 공유하는 기본 좌표)
    base_shared_size = m1
    base_shared_coords = perm[:base_shared_size]
    current_pos = base_shared_size

    # 3. 각 클래스별로 J_c 생성
    J_c_list = []
    for c in range(num_classes):
        # Shared coords에서 랜덤하게 일부 제거 (0% ~ 20% 제거)
        drop_ratio = np.random.uniform(0, 0.2)
        num_drop = int(len(base_shared_coords) * drop_ratio)
        if num_drop > 0:
            drop_indices = np.random.choice(len(base_shared_coords), num_drop, replace=False)
            class_shared_coords = [coord for i, coord in enumerate(base_shared_coords) if i not in drop_indices]
        else:
            class_shared_coords = base_shared_coords.copy()
        
        # Unique size에 양수 노이즈 추가 (기본 크기의 0% ~ 20% 추가)
        base_unique_size = m2 - len(class_shared_coords)
        noise = int(m2 * np.random.uniform(0, 0.2))
        unique_size = base_unique_size + noise
        
        # 남은 좌표에서 unique coords 할당
        if current_pos + unique_size <= m:
            unique_coords = perm[current_pos:current_pos + unique_size]
            current_pos += unique_size
        else:
            print("m should be increased")
            # 남은 좌표가 부족하면 가능한 만큼만 사용
            unique_coords = perm[current_pos:]
            current_pos = m
        
        # 해당 클래스의 J_c = (변형된 shared) + unique
        J_c = torch.tensor(class_shared_coords + unique_coords, dtype=torch.long)
        J_c_list.append(J_c)


    # Assign class labels
    if class_distribution is None:
        class_distribution = torch.ones(num_classes) / num_classes
    y = torch.multinomial(class_distribution, n, replacement=True)

    # Process Bernoulli params
    if bernoulli_params is None:
        base_params = 0.3 + 0.4 * torch.rand(1, m)
        class_specific = 0.3 + 0.4 * torch.rand(num_classes, m)
        p_matrix = (1 - class_feature_similarity) * class_specific + class_feature_similarity * base_params.expand(num_classes, m)
    elif isinstance(bernoulli_params, (int, float)):
        base = float(bernoulli_params)
        variation = 0.2 * (1 - class_feature_similarity)
        p_matrix = base + variation * (torch.rand(num_classes, m) - 0.5)
        p_matrix = p_matrix.clamp(0.1, 0.9)
    else:
        p_matrix = bernoulli_params.clone()
    p_matrix = p_matrix.clamp(0.05, 0.95)

    # Generate features
    x = torch.zeros(n, m)
    for c in range(num_classes):
        class_mask = (y == c)
        n_c = class_mask.sum().item()
        if n_c == 0:
            continue
        J_c = J_c_list[c]
        for j in J_c:
            p_cj = p_matrix[c, j].item()
            x[class_mask, j] = torch.bernoulli(torch.full((n_c,), p_cj))

    # Apply feature modifications
    if feature_noise_std > 0:
        x = x + torch.randn_like(x) * feature_noise_std
    if feature_flip_prob > 0:
        flip_mask = torch.rand_like(x) < feature_flip_prob
        x = torch.where(flip_mask, 1 - x, x)
    if uninformative_feature_ratio > 0:
        num_uninformative = int(m * uninformative_feature_ratio)
        uninformative_indices = torch.randperm(m)[:num_uninformative]
        x[:, uninformative_indices] = torch.rand(n, num_uninformative)

    # Store original labels
    y_original = y.clone()
    if label_noise_ratio > 0:
        num_noisy = int(n * label_noise_ratio)
        noisy_indices = torch.randperm(n)[:num_noisy]
        for idx in noisy_indices:
            current_label = y[idx].item()
            new_label = torch.randint(0, num_classes, (1,)).item()
            while new_label == current_label and num_classes > 1:
                new_label = torch.randint(0, num_classes, (1,)).item()
            y[idx] = new_label

    # Generate graph with duplicate checking
    num_homo_edges = int(num_edges * homophily_ratio)
    num_hetero_edges = num_edges - num_homo_edges
    
    # 전체 edge를 저장할 set (undirected로 저장: min, max 순서)
    edge_set = set()
    
    class_nodes = [torch.where(y_original == c)[0] for c in range(num_classes)]

    # Homophilic edges (중복 체크하면서 추가)
    if num_homo_edges > 0:
        valid_classes = [c for c in range(num_classes) if len(class_nodes[c]) >= 2]
        attempts = 0
        max_attempts = num_homo_edges * 20  # 충분한 시도 횟수
        
        while len([e for e in edge_set if y_original[e[0]] == y_original[e[1]]]) < num_homo_edges and attempts < max_attempts:
            c = valid_classes[torch.randint(len(valid_classes), (1,)).item()]
            nodes = class_nodes[c]
            if len(nodes) >= 2:
                idx = torch.randperm(len(nodes))[:2]
                u, v = nodes[idx[0]].item(), nodes[idx[1]].item()
                if u != v:
                    # undirected edge로 저장 (작은 노드 번호가 앞에)
                    edge = (min(u, v), max(u, v))
                    if edge not in edge_set:
                        edge_set.add(edge)
            attempts += 1

    # Heterophilic edges (중복 체크하면서 추가)
    if num_hetero_edges > 0:
        attempts = 0
        max_attempts = num_hetero_edges * 20
        
        while len([e for e in edge_set if y_original[e[0]] != y_original[e[1]]]) < num_hetero_edges and attempts < max_attempts:
            c1, c2 = torch.randperm(num_classes)[:2].tolist()
            if len(class_nodes[c1]) > 0 and len(class_nodes[c2]) > 0:
                u = class_nodes[c1][torch.randint(len(class_nodes[c1]), (1,))].item()
                v = class_nodes[c2][torch.randint(len(class_nodes[c2]), (1,))].item()
                edge = (min(u, v), max(u, v))
                if edge not in edge_set:
                    edge_set.add(edge)
            attempts += 1

    # edge_set을 edge_index로 변환 (양방향)
    if edge_set:
        edges = list(edge_set)
        # undirected: 양방향 추가
        edge_index = torch.tensor(
            [[e[0] for e in edges] + [e[1] for e in edges],
             [e[1] for e in edges] + [e[0] for e in edges]],
            dtype=torch.long
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Add self-loops
    if add_self_loops:
        self_loops = torch.arange(n).unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, self_loops], dim=1)

    # Generate masks
    indices = torch.randperm(n)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[indices[:n_train]] = True
    val_mask[indices[n_train:n_train + n_valid]] = True
    test_mask[indices[n_train + n_valid:]] = True

    data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.num_classes = num_classes
    data.homophily_ratio = homophily_ratio
    return data


def create_random_split(n, train_ratio=0.6, valid_ratio=0.15, test_ratio=0.25, seed=None):
    """주어진 seed로 랜덤 split 생성"""
    if seed is not None:
        rng = np.random.RandomState(seed)
        indices = torch.from_numpy(rng.permutation(n))
    else:
        indices = torch.randperm(n)

    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[indices[:n_train]] = True
    val_mask[indices[n_train:n_train + n_valid]] = True
    test_mask[indices[n_train + n_valid:]] = True

    return train_mask, val_mask, test_mask


def compute_homophily(data: Data) -> float:
    edge_index = data.edge_index
    y = data.y
    same_label = (y[edge_index[0]] == y[edge_index[1]]).float()
    return same_label.mean().item()
