# In-Run Data Shapley for Federated Learning: 코드 구현 요청 프롬프트

## 프로젝트 개요

In-Run Data Shapley (ICLR 2025) 방법론을 Federated Learning 환경에 적용하여, FL 학습 과정에서 각 클라이언트의 기여도를 Shapley value 기반으로 평가하는 실험 프레임워크를 구현합니다.

---

## 핵심 아이디어

### 원 논문의 방법
- In-Run Data Shapley는 학습의 매 iteration마다 "local utility function"을 정의하고, Taylor 근사를 통해 Shapley value를 분석적으로 계산합니다.
- 1차 근사: `ϕ_z(U^(t)) = -η * ∇ℓ(w_t, z_val) · ∇ℓ(w_t, z)` (gradient dot-product)
- 2차 근사: 1차 항 + `(η²/2) * ∇ℓ(w_t, z)ᵀ H^(z_val) Σ_{z_j∈B_t} ∇ℓ(w_t, z_j)` (gradient-Hessian-gradient product)

### FL 환경으로의 매핑
- **Player**: 각 클라이언트 (원 논문에서의 data point에 대응)
- **Iteration**: 각 FL 라운드의 서버 aggregation 시점 (원 논문에서의 학습 iteration에 대응)
- **Utility function**: `U^(t)(S) = ℓ(w_t - η * Σ_{c∈S} Δw_c^(t), z_val) - ℓ(w_t, z_val)`
  - 여기서 `Δw_c^(t)`는 클라이언트 c의 로컬 학습 완료 후 모델 업데이트
  - S ⊆ C_t는 해당 라운드 참여 클라이언트의 부분집합

### 1차 In-Run Data Shapley (FL 버전)
```
ϕ_c(U^(t)_(1)) = -η * ∇ℓ(w_t, z_val) · Δw_c^(t)
```
- 서버가 validation gradient와 각 클라이언트 업데이트의 내적을 계산

### 2차 In-Run Data Shapley (FL 버전)
```
ϕ_c(U^(t)) = -η * ∇ℓ(w_t, z_val) · Δw_c^(t) + (η²/2) * Δw_c^(t)ᵀ H^(z_val) Σ_{c'∈C_t} Δw_{c'}^(t)
```
- 추가로 gradient-Hessian-gradient product 항이 클라이언트 간 상호작용을 포착

### 최종 기여도
```
ϕ_c = Σ_{t: c ∈ C_t} ϕ_c(U^(t))
```
- 클라이언트가 참여한 모든 라운드의 기여값을 누적

---

## 프로젝트 구조

```
fed_shapley/
├── README.md                  # 프로젝트 설명, 환경 설정, 파일 역할, 실험 재현 가이드
├── main.py                    # 실험 진입점 (args 파싱 및 실험 실행)
├── config.py                  # argument 정의 (argparse)
├── data/
│   ├── datasets.py            # 데이터셋 로딩 (CIFAR10, CIFAR100, TinyImageNet)
│   └── partition.py           # FL 데이터 분할 (IID, Non-IID Dirichlet, Quantity skew)
├── models/
│   └── resnet.py              # ResNet-18 모델 정의
├── fl/
│   ├── server.py              # FL 서버 (aggregation, global model 관리)
│   ├── client.py              # FL 클라이언트 (로컬 학습)
│   └── trainer.py             # FL 학습 루프 (라운드 반복)
├── shapley/
│   ├── in_run_shapley.py      # In-Run Data Shapley 계산 (1차, 2차)
│   ├── exact_shapley.py       # Exact Shapley value 계산 (ground truth, 클라이언트 ≤ 10)
│   └── mc_shapley.py          # Monte Carlo Shapley 추정 (ground truth, 클라이언트 > 10)
├── centralized/
│   └── centralized_trainer.py # Centralized 학습 + data-point level In-Run Data Shapley
├── eval/
│   ├── fidelity.py            # Fidelity 평가 (RMSE, Spearman correlation)
│   ├── client_removal.py      # Client removal experiment
│   └── noisy_client.py        # Noisy client detection (AUROC)
├── utils/
│   ├── logger.py              # 실험 로깅 (매 라운드 기여값, 성능 등)
│   ├── seed.py                # 시드 고정
│   └── visualize.py           # 결과 시각화
└── scripts/
    ├── run_basic.sh            # 기본 실험 스크립트
    ├── run_noniid.sh           # Non-IID 실험 스크립트
    ├── run_k_sweep.sh          # K sweep 실험 스크립트
    ├── run_grid_search.py      # 여러 args 조합에 대해 자동으로 실험 실행 (grid search)
    └── grid_configs/           # grid search용 YAML 설정 파일 디렉터리
        ├── alpha_sweep.yaml    # Dirichlet alpha sweep 설정
        ├── k_sweep.yaml        # local_epochs(K) sweep 설정
        └── noise_sweep.yaml    # 노이즈 클라이언트 실험 sweep 설정
```

---

## 상세 구현 명세

### 1. config.py (Argument 정의)

```python
# 필수 args 목록:

# -- 실험 기본 설정 --
--exp_name: str          # 실험 이름
--seed: int              # 랜덤 시드 (기본값: 42)
--device: str            # cuda or cpu
--output_dir: str        # 결과 저장 경로

# -- 데이터 설정 --
--dataset: str           # cifar10, cifar100, tinyimagenet (기본값: cifar10)
--data_dir: str          # 데이터 저장 경로

# -- 모델 설정 --
--model: str             # resnet18 (기본값: resnet18)

# -- FL 설정 --
--num_clients: int       # 전체 클라이언트 수 (기본값: 10)
--clients_per_round: int # 라운드당 참여 클라이언트 수 (기본값: num_clients, 즉 full participation)
--num_rounds: int        # 총 FL 라운드 수 (기본값: 100)
--local_epochs: int      # 클라이언트 로컬 epoch 수 (기본값: 5)
--local_lr: float        # 클라이언트 로컬 학습률 (기본값: 0.01)
--local_batch_size: int  # 클라이언트 로컬 배치 크기 (기본값: 64)
--aggregation: str       # fedavg (기본값: fedavg)

# -- 데이터 분할 설정 --
--partition: str         # iid, dirichlet (기본값: iid)
--dirichlet_alpha: float # Dirichlet 분포 concentration parameter (기본값: 0.5)
--quantity_skew: bool    # 데이터 수 불균형 여부 (기본값: False)
--quantity_beta: float   # 불균등 시 Dirichlet로 데이터 수 배분 파라미터 (기본값: 0.5)

# -- Shapley 설정 --
--use_second_order: bool # 2차 항 사용 여부 (기본값: True)
--num_val_samples: int   # validation 샘플 수 (기본값: 500)
--mc_permutations: int   # Monte Carlo permutation 수 (기본값: 1000)

# -- 평가 설정 --
--eval_every: int        # 몇 라운드마다 평가할지 (기본값: 1)
--noisy_clients: list    # 노이즈를 주입할 클라이언트 인덱스 (기본값: [])
--noise_type: str        # label_flip, random_update (기본값: label_flip)
--noise_ratio: float     # 라벨 뒤집기 비율 (기본값: 0.5)

# -- 비교 실험 설정 --
--run_centralized: bool  # centralized 비교 실험 수행 여부 (기본값: False)
--run_exact_shapley: bool # exact shapley ground truth 계산 여부 (기본값: True)

# -- 로깅 설정 --
--use_wandb: bool         # WandB 로깅 사용 여부 (기본값: False)
--wandb_project: str      # WandB 프로젝트 이름 (기본값: "fed_shapley")
--wandb_run_name: str     # WandB run 이름 (기본값: exp_name 자동 사용)
--wandb_entity: str       # WandB entity/team 이름 (기본값: None)
--use_tensorboard: bool   # TensorBoard 로깅 사용 여부 (기본값: False)
--log_every: int          # 몇 라운드마다 로깅할지 (기본값: 1)
```

### 2. data/partition.py (데이터 분할)

```
구현해야 할 함수:

partition_data(dataset, num_clients, partition_type, **kwargs) -> dict[int, list[int]]
  - 반환값: {client_id: [sample_indices]} 매핑

1. IID 분할:
   - 전체 데이터를 균등하게 랜덤 분할

2. Dirichlet Non-IID 분할:
   - 각 클래스에 대해 Dirichlet(alpha) 분포로 클라이언트별 비율 생성
   - alpha 값에 따라 이질성 제어 (0.1: 극심한 non-IID, 100: 거의 IID)

3. Quantity skew:
   - Dirichlet(beta) 분포로 클라이언트별 데이터 수 비율 생성
   - label 분포는 IID 또는 Non-IID와 독립적으로 조합 가능

데이터 분할 결과를 시각화하는 함수도 포함 (각 클라이언트의 클래스 분포 히스토그램)
```

### 3. fl/server.py (서버)

```python
class Server:
    def __init__(self, model, val_loader, device, args):
        self.global_model = model
        self.val_loader = val_loader  # 서버가 보유하는 validation set

    def aggregate(self, client_updates, client_weights):
        """
        FedAvg aggregation
        - 균등 데이터: 단순 평균 (가중치 동일)
        - 불균등 데이터: 데이터 수 비례 가중 평균
        client_updates: list of state_dict diffs (Δw_c)
        client_weights: list of float (데이터 수 비례 가중치 또는 균등 가중치)
        """

    def compute_validation_gradient(self):
        """
        글로벌 모델 w_t에서 validation set에 대한 gradient 계산
        반환: flattened gradient vector ∇ℓ(w_t, D_val)
        """

    def compute_validation_hessian_vector_product(self, vector):
        """
        H^(z_val) @ vector 계산
        torch.autograd를 활용한 효율적 Hessian-vector product
        """
```

### 4. fl/client.py (클라이언트)

```python
class Client:
    def __init__(self, client_id, train_loader, device, args):
        self.client_id = client_id
        self.train_loader = train_loader

    def local_train(self, global_model_state):
        """
        글로벌 모델을 받아 로컬 학습 수행
        반환: Δw_c = w_local - w_global (state_dict diff)
        """
```

### 5. fl/trainer.py (FL 학습 루프)

```python
class FLTrainer:
    def __init__(self, server, clients, shapley_calculator, args):
        pass

    def train(self):
        """
        메인 FL 학습 루프:
        for round_t in range(num_rounds):
            1. 클라이언트 선택 (full 또는 partial participation)
            2. 선택된 클라이언트에 글로벌 모델 배포
            3. 각 클라이언트 로컬 학습 → Δw_c 수집
            4. In-Run Data Shapley 계산 (서버에서)
               - validation gradient 계산
               - 각 Δw_c와의 내적 (1차 항)
               - Hessian-vector product (2차 항, 옵션)
            5. Aggregation (FedAvg)
            6. 글로벌 모델 업데이트
            7. 평가 및 로깅

        매 라운드의 기여값, 글로벌 모델 성능 등을 기록
        """
```

### 6. shapley/in_run_shapley.py (핵심 구현)

```python
class InRunDataShapley:
    def __init__(self, server, args):
        self.use_second_order = args.use_second_order

    def compute_round_shapley(self, client_updates, client_ids, round_idx):
        """
        한 라운드에서의 In-Run Data Shapley 계산

        Args:
            client_updates: list of flattened Δw_c vectors
            client_ids: list of participating client ids
            round_idx: 현재 라운드 번호

        Returns:
            dict {client_id: shapley_value}

        구현:
        1) val_grad = server.compute_validation_gradient()  # ∇ℓ(w_t, z_val)

        2) 1차 항 계산:
           for each client c:
               first_order[c] = -η * dot(val_grad, Δw_c)

        3) 2차 항 계산 (use_second_order=True인 경우):
           aggregated_update = Σ Δw_c  # 모든 참여 클라이언트 업데이트 합
           hvp = server.compute_validation_hessian_vector_product(aggregated_update)
           for each client c:
               second_order[c] = (η² / 2) * dot(Δw_c, hvp)

        4) 최종: shapley[c] = first_order[c] + second_order[c]
        """

    def accumulate(self, round_shapley_values):
        """
        라운드별 기여값을 누적하여 최종 기여도 산출
        """
```

### 7. shapley/exact_shapley.py (Ground Truth)

```python
class ExactShapley:
    def __init__(self, server, args):
        pass

    def compute_round_exact_shapley(self, client_updates, client_ids, round_idx):
        """
        한 라운드에서의 exact Shapley value 계산 (클라이언트 ≤ 10)

        모든 가능한 부분집합 S ⊆ C_t에 대해:
        1) aggregated_update(S) = Σ_{c∈S} Δw_c
        2) w_new(S) = w_t - η * aggregated_update(S)
        3) U(S) = ℓ(w_new(S), D_val) - ℓ(w_t, D_val)

        Shapley formula:
        ϕ_c = (1/n) * Σ_{k=1}^{n} C(n-1,k-1)^{-1} * Σ_{S⊆C\{c}, |S|=k-1} [U(S∪{c}) - U(S)]

        반환: dict {client_id: exact_shapley_value}
        """
```

### 8. shapley/mc_shapley.py (Monte Carlo Ground Truth)

```python
class MonteCarloShapley:
    def __init__(self, server, args):
        self.num_permutations = args.mc_permutations  # 기본값 1000

    def compute_round_mc_shapley(self, client_updates, client_ids, round_idx):
        """
        Monte Carlo 추정 (클라이언트 > 10)

        for each permutation π:
            for i in range(len(π)):
                S_before = π[:i]의 클라이언트 업데이트 합으로 모델 구성
                S_after = π[:i+1]의 클라이언트 업데이트 합으로 모델 구성
                marginal[π[i]] += U(S_after) - U(S_before)

        ϕ_c = marginal[c] / num_permutations

        반환: dict {client_id: mc_shapley_value}
        """
```

### 9. centralized/centralized_trainer.py

```python
class CentralizedTrainer:
    def __init__(self, model, train_loader, val_loader, client_data_indices, args):
        """
        client_data_indices: FL과 동일한 데이터 분할 정보
        centralized 학습을 하되, 각 data point가 어떤 client에 속하는지 추적
        """

    def train_and_compute_shapley(self):
        """
        일반적인 centralized SGD 학습을 수행하면서,
        매 iteration마다:
        1) validation gradient 계산
        2) 배치 내 각 data point에 대한 gradient dot-product 계산
        3) 각 data point의 기여값을 해당 client_id에 누적

        반환: dict {client_id: accumulated_shapley_value}

        이것이 "FL 구조가 없었다면 얻을 수 있는 기여도"로,
        FL 환경의 기여도와 비교하는 baseline이 됨
        """
```

### 10. eval/fidelity.py

```python
def compute_fidelity(in_run_values, ground_truth_values):
    """
    In-Run Data Shapley와 ground truth (exact/MC) 비교

    Metrics:
    - RMSE: Root Mean Squared Error
    - Spearman rank correlation: 순위 상관
    - Pearson correlation: 선형 상관

    원 논문의 Figure 1 스타일 scatter plot 생성
    """
```

### 11. eval/client_removal.py

```python
def client_removal_experiment(server, clients, shapley_values, args):
    """
    기여값 기반 클라이언트 제거 실험

    1) shapley_values 기준으로 클라이언트 정렬
    2) 가장 기여값이 높은 순으로 10%, 20%, ..., 100% 제거
    3) 남은 클라이언트로 FL 학습 재수행
    4) 각 단계의 글로벌 모델 테스트 정확도 기록

    기여값이 낮은 순으로 제거하는 실험도 동시 수행
    랜덤 제거 baseline도 포함

    결과: 제거 비율 vs 테스트 정확도 그래프
    """
```

### 12. eval/noisy_client.py

```python
def noisy_client_detection(shapley_values, noisy_client_ids, all_client_ids):
    """
    노이즈 클라이언트 탐지 성능 평가

    - 일부 클라이언트에 label flipping 적용 (args.noisy_clients, args.noise_ratio)
    - 기여값이 낮을수록 노이즈 클라이언트일 확률이 높다고 판단
    - AUROC 계산: 기여값으로 noisy vs clean 클라이언트 분류 성능

    반환: AUROC score
    """
```

### 13. utils/logger.py

```python
class ExperimentLogger:
    """
    매 라운드 기록:
    - round_idx
    - participating_client_ids
    - per_client_in_run_shapley (1차, 2차 각각)
    - per_client_exact_or_mc_shapley (ground truth)
    - global_model_val_loss
    - global_model_val_accuracy
    - global_model_test_accuracy
    - cumulative_shapley_values

    전체 실험 종료 후:
    - 최종 누적 기여값
    - fidelity metrics (RMSE, Spearman)
    - client removal 결과
    - noisy client detection AUROC

    저장 형식: JSON 또는 CSV + 시각화 이미지
    """
```

### 14. 학습 진행 과정 및 결과 저장 — WandB / TensorBoard 활용

WandB를 기본 로깅 백엔드로 사용하고, TensorBoard를 선택적 폴백으로 지원한다.
GhostSuite의 `requirements.txt`에 `wandb==0.23.1`, `tensorboard==2.20.0`이 이미 포함되어 있으므로 추가 설치 없이 사용 가능하다.

```python
# utils/logger.py 내 WandB/TensorBoard 통합 구현 명세

class ExperimentLogger:
    def __init__(self, args):
        # WandB 초기화
        if args.use_wandb:
            import wandb
            wandb.init(
                project=args.wandb_project,      # 예: "fed_shapley"
                name=args.wandb_run_name or args.exp_name,
                entity=args.wandb_entity,
                config=vars(args),               # 모든 하이퍼파라미터 자동 기록
                dir=args.output_dir,
            )
            self.wandb = wandb

        # TensorBoard 초기화
        if args.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=args.output_dir)

    def log_round(self, round_idx: int, metrics: dict):
        """
        매 라운드마다 호출. metrics 딕셔너리 키 예시:
          - "val/loss", "val/accuracy"
          - "test/accuracy"
          - "shapley/client_{i}_inrun_1st"
          - "shapley/client_{i}_inrun_2nd"
          - "shapley/client_{i}_exact"
          - "shapley/client_{i}_cumulative"
          - "train/round_time_sec"
        """
        if hasattr(self, "wandb"):
            self.wandb.log(metrics, step=round_idx)
        if hasattr(self, "tb_writer"):
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, global_step=round_idx)

    def log_final_summary(self, summary: dict):
        """
        실험 종료 시 호출. summary 딕셔너리 키 예시:
          - "final/spearman_corr"
          - "final/rmse"
          - "final/noisy_detection_auroc"
          - "final/best_test_accuracy"
          - 클라이언트 제거 실험 결과 테이블
        """
        if hasattr(self, "wandb"):
            self.wandb.summary.update(summary)
            # 클라이언트 기여도 bar chart
            self.wandb.log({"shapley/final_bar": self.wandb.plot.bar(...)})
        if hasattr(self, "tb_writer"):
            self.tb_writer.add_hparams(hparam_dict=..., metric_dict=summary)

    def save_to_disk(self, output_dir: str):
        """
        WandB/TensorBoard와 별개로 로컬에도 저장:
        - {output_dir}/results.json    : 전체 라운드별 metrics
        - {output_dir}/shapley.csv     : 클라이언트별 누적 기여값 테이블
        - {output_dir}/config.json     : args 스냅샷
        """

    def finish(self):
        if hasattr(self, "wandb"):
            self.wandb.finish()
        if hasattr(self, "tb_writer"):
            self.tb_writer.close()
```

**WandB 활용 시 권장 실행 방식:**
```bash
# 단일 실험
python main.py --use_wandb True --wandb_project fed_shapley \
    --exp_name "iid_k5_c10" --num_clients 10 ...

# WandB sweep (wandb agent 방식)
wandb sweep sweep_config.yaml
wandb agent <sweep_id>
```

**TensorBoard 활용 시:**
```bash
python main.py --use_tensorboard True --output_dir ./runs/exp1 ...
tensorboard --logdir ./runs
```

---

### 15. scripts/run_grid_search.py — Grid Search 자동화

여러 args 조합에 대해 자동으로 실험을 실행하는 Python 스크립트.
bash 루프 대신 Python으로 구현하여 결과 수집·요약까지 한 파일에서 처리한다.

```python
# scripts/run_grid_search.py 구현 명세
"""
사용 예시:
  python scripts/run_grid_search.py --config scripts/grid_configs/alpha_sweep.yaml
  python scripts/run_grid_search.py --config scripts/grid_configs/k_sweep.yaml --dry_run
  python scripts/run_grid_search.py --config scripts/grid_configs/k_sweep.yaml --max_runs 5
"""

import itertools
import subprocess
import yaml
import csv
import os
import json
from pathlib import Path
from datetime import datetime

# YAML 설정 파일 예시 (scripts/grid_configs/alpha_sweep.yaml):
# base_args:
#   dataset: cifar10
#   num_clients: 10
#   num_rounds: 100
#   local_epochs: 5
#   run_exact_shapley: True
#   use_second_order: True
#   use_wandb: True
#   wandb_project: fed_shapley
# grid:
#   partition: [dirichlet]
#   dirichlet_alpha: [0.1, 0.5, 1.0, 5.0, 100.0]
#   seed: [42, 123, 456]

def load_grid_config(config_path: str) -> dict:
    """YAML 설정 파일 로드."""

def generate_experiments(base_args: dict, grid: dict) -> list[dict]:
    """
    grid의 Cartesian product를 생성하여 실험 args 리스트 반환.
    itertools.product 사용.
    예: grid = {a: [1,2], b: [x,y]} → [{a:1,b:x}, {a:1,b:y}, {a:2,b:x}, {a:2,b:y}]
    """

def args_to_cmd(args: dict, output_dir: str) -> list[str]:
    """
    args 딕셔너리를 python main.py 커맨드 리스트로 변환.
    실험별 output_dir을 args 조합으로 자동 생성.
    예: ./outputs/grid/alpha0.1_seed42/
    """

def run_experiment(cmd: list[str], timeout: int = None) -> dict:
    """
    subprocess.run으로 실험 실행.
    완료 후 output_dir/results.json에서 최종 메트릭 읽어 반환.
    실패 시 에러 정보 반환.
    """

def save_summary_csv(results: list[dict], output_path: str):
    """
    모든 실험 결과를 하나의 CSV 파일로 저장.
    컬럼: 모든 grid args + 결과 메트릭(val_acc, spearman, rmse, auroc 등)
    """

def main():
    """
    CLI args: --config, --dry_run, --max_runs, --output_root, --timeout
    1. YAML 로드 → 실험 리스트 생성
    2. --dry_run이면 실험 커맨드만 출력하고 종료
    3. 순차 실행 (각 실험은 독립 프로세스)
    4. 완료 후 summary CSV 저장: {output_root}/grid_results_{timestamp}.csv
    5. 간단한 결과 요약 출력 (best run, failed runs 등)
    """
```

---

### 16. README.md 구성 명세

프로젝트 루트에 `README.md`를 작성한다. 다음 섹션을 모두 포함해야 한다:

```
# FL In-Run Data Shapley

## 개요
- 프로젝트 목적: In-Run Data Shapley (ICLR 2025)를 FL 환경에 적용하여 클라이언트 기여도 평가
- 핵심 수식 요약 (1차, 2차 Shapley)
- GhostSuite와의 관계 (본 프레임워크는 GhostSuite와 독립적으로 동작)

## 환경 요구사항
- Python >= 3.10
- PyTorch >= 2.0 (CUDA 11.8+ 권장)
- 주요 패키지: torchvision, wandb, tensorboard, scikit-learn, pyyaml, tqdm

## 설치
pip install -r requirements.txt

## 프로젝트 구조
각 파일/디렉터리의 역할을 한 줄 설명:
  - main.py: 실험 진입점
  - config.py: 하이퍼파라미터 정의 (전체 args 목록)
  - data/: 데이터셋 로딩 및 FL 분할 (IID, Dirichlet Non-IID, Quantity skew)
  - models/: ResNet-18 정의
  - fl/: FL 학습 루프 (Server, Client, FLTrainer)
  - shapley/: In-Run / Exact / MC Shapley 계산
  - centralized/: Centralized 비교 실험
  - eval/: Fidelity 평가, Client removal, Noisy client detection
  - utils/: 로깅, 시드 고정, 시각화
  - scripts/: 실험 자동화 (grid search, sweep)

## 빠른 시작
# 최소 실험 (클라이언트 5개, IID, CIFAR-10)
python main.py --num_clients 5 --partition iid --num_rounds 50 \
    --local_epochs 5 --run_exact_shapley True --exp_name quick_test

## 주요 실험 재현
각 실험 시나리오별 커맨드 (실험 1~6 요약)
scripts/ 디렉터리의 grid search 활용 방법

## 결과 해석
- shapley.csv: 각 클라이언트의 누적 기여값. 양수 = 도움, 음수 = 해로움
- Spearman correlation: In-Run Shapley vs Exact Shapley 순위 일치도 (1에 가까울수록 좋음)
- AUROC: 노이즈 클라이언트 탐지 성능 (0.5 = 랜덤, 1.0 = 완벽)
- Client removal curve: 기여값 높은 순 제거 시 정확도 하락 빨라야 함

## 재현성
- 모든 실험은 --seed로 시드 고정
- 최소 3개 시드(42, 123, 456)로 반복 후 평균 ± 표준편차 보고 권장
```

---

### 17. 전체 코드 주석 가이드라인

모든 `.py` 파일에 다음 기준으로 주석과 docstring을 작성한다.

```
1. 파일 상단 모듈 docstring
   - 파일 목적 한 줄 요약
   - 구현하는 논문 수식 번호 (예: "Eq. (3) in In-Run Data Shapley, ICLR 2025")
   - 주요 입출력 타입 요약
   예시:
   \"\"\"
   shapley/in_run_shapley.py
   목적: 1차 및 2차 In-Run Data Shapley 계산 (논문 Eq. 3, 5)
   입력: client_updates (List[Tensor]), val_grad (Tensor)
   출력: {client_id: shapley_value} 딕셔너리
   \"\"\"

2. 클래스/함수 docstring (Google style)
   def compute_round_shapley(self, client_updates, client_ids, round_idx):
       \"\"\"
       한 FL 라운드의 In-Run Data Shapley 계산.

       Args:
           client_updates: 각 클라이언트의 flattened Δw_c (List[Tensor], shape [num_params])
           client_ids: 참여 클라이언트 ID 목록 (List[int])
           round_idx: 현재 라운드 번호 (int, 0-indexed)

       Returns:
           dict[int, float]: {client_id: shapley_value}

       Note:
           1차 항: ϕ_c^(1) = -η * <∇ℓ(w_t, z_val), Δw_c>  (논문 Eq. 3)
           2차 항: ϕ_c^(2) = (η²/2) * Δw_c^T H Σ Δw_{c'}   (논문 Eq. 5)
       \"\"\"

3. 수식 구현부 인라인 주석
   # ∇ℓ(w_t, D_val) — validation gradient, shape: [num_params]
   val_grad = self.server.compute_validation_gradient()

   # 1차 항: ϕ_c^(1) = -η * dot(val_grad, Δw_c)
   first_order = {cid: -eta * torch.dot(val_grad, dw)
                  for cid, dw in zip(client_ids, client_updates)}

   # 집계된 클라이언트 업데이트 합: Σ_{c ∈ C_t} Δw_c
   aggregated = sum(client_updates)

   # H @ aggregated — Hessian-vector product (논문 Eq. 5의 H^(z_val) 항)
   hvp = self.server.compute_validation_hessian_vector_product(aggregated)

4. Type hint 필수 적용
   - 함수 인자와 반환값에 모두 type hint 작성
   - 복잡한 타입은 typing 모듈 활용 (List, Dict, Optional, Tuple 등)

5. TODO / NOTE 주석 활용
   # NOTE: 클라이언트 100개 이상일 때 메모리 > 4GB 가능 → 디스크 저장 권장
   # TODO Phase 3: Dirichlet Non-IID 분할 추가 예정
```

---

## 실험 시나리오

### 실험 1: Fidelity 평가 (기본)
```bash
# 클라이언트 5개, IID, full participation
python main.py --num_clients 5 --partition iid --clients_per_round 5 \
    --num_rounds 100 --local_epochs 5 --run_exact_shapley True \
    --use_second_order True --dataset cifar10

# 클라이언트 10개
python main.py --num_clients 10 --partition iid --clients_per_round 10 \
    --num_rounds 100 --local_epochs 5 --run_exact_shapley True \
    --use_second_order True --dataset cifar10

# 클라이언트 100개 (Monte Carlo)
python main.py --num_clients 100 --partition iid --clients_per_round 100 \
    --num_rounds 100 --local_epochs 5 --run_exact_shapley False \
    --mc_permutations 1000 --use_second_order True --dataset cifar10
```

### 실험 2: K sweep (통신 빈도 vs 정확도 trade-off)
```bash
# 총 학습량을 동일하게 유지하면서 local_epochs 변화
# 예: 총 500 epoch 기준
for K in 1 2 5 10 20 50; do
    ROUNDS=$((500 / K))
    python main.py --num_clients 10 --local_epochs $K --num_rounds $ROUNDS \
        --partition iid --run_exact_shapley True --dataset cifar10
done
```

### 실험 3: Non-IID 영향
```bash
# Dirichlet alpha 변화
for ALPHA in 0.1 0.5 1.0 5.0 100.0; do
    python main.py --num_clients 10 --partition dirichlet \
        --dirichlet_alpha $ALPHA --num_rounds 100 --local_epochs 5 \
        --run_exact_shapley True --dataset cifar10
done
```

### 실험 4: Quantity skew 영향
```bash
python main.py --num_clients 10 --partition iid --quantity_skew True \
    --quantity_beta 0.5 --num_rounds 100 --local_epochs 5 \
    --run_exact_shapley True --dataset cifar10
```

### 실험 5: Noisy client detection
```bash
# 클라이언트 10개 중 2개에 label flipping
python main.py --num_clients 10 --partition iid \
    --noisy_clients 0 1 --noise_type label_flip --noise_ratio 0.5 \
    --num_rounds 100 --local_epochs 5 --run_exact_shapley True \
    --dataset cifar10
```

### 실험 6: Centralized 비교
```bash
python main.py --num_clients 10 --partition iid --num_rounds 100 \
    --local_epochs 5 --run_centralized True --run_exact_shapley True \
    --dataset cifar10
```

---

## 구현 우선순위

### Phase 1: 기본 FL + 1차 In-Run Data Shapley
1. config.py, seed.py
2. datasets.py, partition.py (IID만 먼저)
3. resnet.py
4. client.py, server.py (validation gradient 계산 포함)
5. in_run_shapley.py (1차 항만)
6. exact_shapley.py
7. trainer.py
8. fidelity.py
9. logger.py
10. main.py

→ **검증**: 클라이언트 5개, IID, CIFAR10에서 1차 In-Run Shapley와 Exact Shapley 비교

### Phase 2: 2차 항 추가
1. server.py에 Hessian-vector product 추가
2. in_run_shapley.py에 2차 항 추가

→ **검증**: 1차 vs 2차의 정확도 차이 확인

### Phase 3: Non-IID 및 Quantity skew
1. partition.py에 Dirichlet, quantity skew 추가
2. 관련 실험 수행

### Phase 4: 평가 도구
1. client_removal.py
2. noisy_client.py
3. visualize.py

### Phase 5: Centralized 비교
1. centralized_trainer.py

### Phase 6: 확장 (CIFAR100, TinyImageNet, 클라이언트 100개)
1. Monte Carlo Shapley 구현
2. 데이터셋 확장
3. 대규모 실험

---

## 핵심 구현 시 주의사항

### 1. 모델 업데이트 Δw_c의 표현
- 클라이언트 로컬 학습 전후의 state_dict 차이로 계산
- 내적 등 연산을 위해 flattened vector로 변환하는 유틸리티 함수 필요
- `state_dict_to_vector(state_dict)`, `vector_to_state_dict(vector, reference_state_dict)` 구현

### 2. FedAvg Aggregation에서 η의 정의
- 균등 데이터: `w_{t+1} = w_t - (1/|C_t|) * Σ Δw_c`, 유효 η = 1/|C_t|
- 불균등 데이터: `w_{t+1} = w_t - Σ (n_c/n_total) * Δw_c`, 유효 η는 클라이언트별로 n_c/n_total
- Shapley 계산 시 이 η를 일관되게 사용해야 함

### 3. Hessian-vector product 계산
```python
# torch.autograd를 활용한 효율적 구현
def hvp(loss, params, vector):
    grad = torch.autograd.grad(loss, params, create_graph=True)
    grad_flat = flatten(grad)
    hvp_result = torch.autograd.grad(grad_flat @ vector, params)
    return flatten(hvp_result)
```

### 4. Exact Shapley에서 subset evaluation
```python
# 각 부분집합 S에 대해:
# 1) 해당 클라이언트 업데이트만 합산
# 2) 글로벌 모델에 적용
# 3) validation loss 계산
# 공집합 S={}일 때는 U({}) = 0 (모델 변화 없음)
```

### 5. 메모리 관리
- 클라이언트 업데이트 벡터가 ResNet-18 기준 약 11M 파라미터 × 4 bytes = 44MB
- 클라이언트 100개면 4.4GB → 메모리 관리 필요
- 필요시 디스크에 저장하고 순차적으로 로드

### 6. 재현성
- 모든 random seed 고정: numpy, torch, random, cuda
- 데이터 분할 seed, 클라이언트 선택 seed, 모델 초기화 seed 분리
- 최소 3개 seed로 실험 반복, 평균 ± 표준편차 보고
