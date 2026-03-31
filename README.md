<style>
    body {
        color: #CDD3DE !important;
    }
    pre, code {
        color: #CDD3DE !important;
    }
</style>
# FedLLM_CE

**Federated Learning 환경에서 In-Run Data Shapley를 활용한 클라이언트 기여도 평가 프레임워크**

ICLR 2025 Outstanding Paper Runner-up ["Data Shapley in One Training Run"](https://openreview.net/pdf?id=HD6bWcj87Y)의 핵심 아이디어를 Federated Learning(FL)에 적용하여, 학습 중 실시간으로 각 클라이언트의 기여도(Shapley Value)를 계산하는 프레임워크입니다.

---

## 목차

1. [핵심 개념](#핵심-개념)
2. [프로젝트 구조](#프로젝트-구조)
3. [설치 방법](#설치-방법)
4. [빠른 시작](#빠른-시작)
5. [상세 사용법](#상세-사용법)
6. [모듈별 상세 설명](#모듈별-상세-설명)
7. [비용 추적 (시간 및 FLOPs)](#비용-추적-시간-및-flops)
8. [실험 스크립트](#실험-스크립트)
9. [실험 결과 요약](#실험-결과-요약)
10. [출력 결과 해석](#출력-결과-해석)
11. [전체 인자(Arguments) 목록](#전체-인자arguments-목록)
12. [수학적 배경](#수학적-배경)

---

## 핵심 개념

### 문제: FL에서 클라이언트 기여도를 어떻게 측정할 것인가?

Federated Learning에서는 여러 클라이언트가 각자의 로컬 데이터로 모델을 학습하고, 그 업데이트를 서버에 보냅니다. 이때 **어떤 클라이언트가 전체 모델 성능에 얼마나 기여했는지**를 정량적으로 측정하는 것이 중요합니다. 이 정보는 다음과 같은 목적에 활용됩니다:

- **데이터 품질 평가**: 유해하거나 노이즈가 있는 클라이언트 탐지
- **공정한 보상 분배**: 기여도에 비례한 인센티브 설계
- **클라이언트 선택**: 다음 라운드에 참여시킬 클라이언트 결정
- **디버깅**: 학습 성능을 해치는 클라이언트 식별

### 해결 방법: In-Run Data Shapley

기존의 Shapley Value 계산은 모든 클라이언트 부분집합($2^n$개)에 대해 모델을 다시 학습해야 하므로 비현실적입니다. 본 프레임워크는 **학습 도중 gradient dot-product만으로** Shapley Value를 근사하여, 추가 학습 없이 기여도를 계산합니다.

**핵심 수식 (1차 근사):**

$$\phi_c^{(t)} = -\eta \cdot \langle \nabla \ell(w_t, D_{\text{val}}), \Delta w_c^{(t)} \rangle$$

- $\phi_c^{(t)}$: 라운드 $t$에서 클라이언트 $c$의 Shapley Value
- $\eta$: 학습률 (learning rate)
- $\nabla \ell(w_t, D_{\text{val}})$: 현재 글로벌 모델의 검증 데이터에 대한 gradient
- $\Delta w_c^{(t)} = w_{\text{local},c} - w_{\text{global}}$: 클라이언트 $c$의 가중치 업데이트

**직관적 해석**: 클라이언트의 업데이트 방향이 검증 손실을 줄이는 방향과 잘 정렬될수록, 해당 클라이언트의 기여도가 높습니다.

---

## 프로젝트 구조

```
FedLLM_CE/
├── CLAUDE.md                    # Claude Code 가이드
├── README.md                    # 이 파일
├── implementation_prompt.md     # 구현 설계 문서
│
├── GhostSuite/                  # 원본 GhostSuite 프레임워크 (참고용)
│
└── fed_shapley/                 # FL Shapley 프레임워크 (메인 코드)
    ├── main.py                  # 실험 진입점 (Entry Point)
    ├── config.py                # CLI 인자 정의 및 파싱
    ├── requirements.txt         # Python 의존성 패키지
    │
    ├── data/                    # 데이터 로딩 및 분할
    │   ├── __init__.py
    │   ├── datasets.py          # CIFAR-10/100, Tiny ImageNet 로더
    │   └── partition.py         # IID, Dirichlet non-IID 파티셔닝
    │
    ├── models/                  # 모델 정의
    │   ├── __init__.py
    │   └── resnet.py            # 데이터셋에 맞게 수정된 ResNet-18
    │
    ├── fl/                      # Federated Learning 핵심 모듈
    │   ├── __init__.py
    │   ├── server.py            # 글로벌 모델 관리, FedAvg 집계, gradient/HVP 계산
    │   ├── client.py            # 로컬 SGD 학습, Δw_c 계산, 노이즈 주입
    │   └── trainer.py           # FL 학습 루프 오케스트레이터
    │
    ├── shapley/                 # Shapley Value 계산 엔진
    │   ├── __init__.py
    │   ├── in_run_shapley.py    # [메인] Gradient dot-product 기반 (효율적)
    │   ├── exact_shapley.py     # 정확한 2^n 부분집합 열거 (n ≤ 10)
    │   └── mc_shapley.py        # Monte Carlo 순열 샘플링 (n > 10)
    │
    ├── centralized/             # 중앙집중 학습 베이스라인
    │   ├── __init__.py
    │   └── centralized_trainer.py  # 샘플 단위 gradient 기여도 → 클라이언트 귀속
    │
    ├── eval/                    # 평가 모듈
    │   ├── __init__.py
    │   ├── fidelity.py          # In-run vs Ground-truth 비교 (RMSE, Pearson, Spearman)
    │   ├── client_removal.py    # 클라이언트 제거 실험
    │   └── noisy_client.py      # 노이즈 클라이언트 탐지 (AUROC)
    │
    ├── utils/                   # 유틸리티
    │   ├── __init__.py
    │   ├── seed.py              # 랜덤 시드 고정
    │   ├── logger.py            # W&B, TensorBoard, 디스크 로깅 (타이밍 CSV 포함)
    │   ├── timer.py             # CostTracker — 벽시계 시간 및 FLOPs 추정
    │   └── visualize.py         # 바 차트, 산점도, 히트맵 시각화
    │
    ├── scripts/                 # 실험 실행 스크립트
    │   ├── run_experiments.sh   # [메인] 3개 실험 통합 실행 (39 runs)
    │   ├── analyze_results.py   # 실험 결과 집계 및 분석 도구
    │   ├── run_basic.sh         # 기본 IID 실험
    │   ├── run_noniid.sh        # Non-IID alpha 스윕
    │   ├── run_k_sweep.sh       # 로컬 에폭 스윕
    │   ├── run_grid_search.py   # YAML 기반 그리드 서치 러너
    │   └── grid_configs/        # 그리드 서치 설정 파일
    │       ├── alpha_sweep.yaml # Dirichlet alpha 스윕
    │       ├── k_sweep.yaml     # 로컬 에폭 스윕
    │       └── noise_sweep.yaml # 노이즈 클라이언트 스윕
    │
    ├── outputs/                 # 실험 결과 저장 디렉토리 (기본)
    │   ├── *.json               # 실험 메트릭 및 설정
    │   ├── *.csv                # Shapley 값 기록, 타이밍 기록
    │   └── figures/             # 시각화 이미지
    │
    └── data_cache/              # 다운로드된 데이터셋 캐시

results/                         # 사전 실행된 실험 결과 (프로젝트 루트)
├── exp1/                        # 실험 1: Non-FL vs FL Shapley 정확도
├── exp2/                        # 실험 2: Communication cost vs accuracy
├── exp3/                        # 실험 3: Non-IID level vs accuracy
└── analysis/                    # 집계된 CSV 요약
```

---

## 설치 방법

### 요구 사항
- Python 3.8+
- PyTorch 2.0+
- CUDA (선택사항, GPU 사용 시)

### 설치

```bash
# 1. 저장소 클론
git clone <repository_url>
cd FedLLM_CE

# 2. 의존성 설치
cd fed_shapley
pip install -r requirements.txt
```

### 주요 의존성 패키지

| 패키지 | 최소 버전 | 용도 |
|--------|----------|------|
| `torch` | 2.0.0 | 모델 학습 및 gradient 계산 |
| `torchvision` | 0.15.0 | 데이터셋 로딩 (CIFAR, ImageNet) |
| `numpy` | 1.24.0 | 수치 연산 |
| `scipy` | 1.11.0 | 통계 검정 (Spearman, Pearson) |
| `scikit-learn` | 1.3.0 | AUROC 등 평가 메트릭 |
| `matplotlib` | 3.7.0 | 시각화 |
| `seaborn` | 0.12.0 | 히트맵 시각화 |
| `wandb` | 0.15.0 | Weights & Biases 로깅 (선택) |
| `tensorboard` | 2.13.0 | TensorBoard 로깅 (선택) |
| `pyyaml` | 6.0 | 그리드 서치 설정 파싱 |
| `tqdm` | 4.65.0 | 진행률 표시 |

---

## 빠른 시작

### 가장 간단한 실행

```bash
cd fed_shapley

# 기본 실험: 10 클라이언트, IID, CIFAR-10, 50 라운드
python main.py \
    --exp_name "quickstart" \
    --num_clients 10 \
    --partition iid \
    --num_rounds 50 \
    --local_epochs 5 \
    --dataset cifar10 \
    --seed 42
```

이 명령은 다음을 수행합니다:
1. CIFAR-10 데이터셋을 다운로드합니다 (최초 실행 시)
2. 학습 데이터를 10개 클라이언트에 균등 분배합니다 (IID)
3. 50 라운드의 FedAvg 학습을 실행합니다
4. 매 라운드마다 In-Run Shapley Value를 계산합니다
5. 결과를 `./outputs/` 디렉토리에 저장합니다

### 실행 결과 예시

```
============================================================
  FedShapley Experiment: quickstart
  Device: cuda
  Dataset: cifar10 | Partition: iid
  Clients: 10 | Rounds: 50
  Second-order: False
============================================================

[Main] Loading dataset ...
[Main] Train size: 50000, Test size: 10000, Classes: 10
[Main] Partitioning data (iid) ...
  Client   0: 4950 samples, 10 classes, dominant=3
  Client   1: 4950 samples, 10 classes, dominant=7
  ...

[FLTrainer] Final Cumulative Shapley Values:
  Rank 1: Client   3 -> phi = +0.042518
  Rank 2: Client   7 -> phi = +0.039214
  ...
  Rank 10: Client  5 -> phi = +0.011023
```

---

## 상세 사용법

### 1. 기본 실험 (IID + Exact Shapley 비교)

```bash
python main.py \
    --exp_name "basic_with_gt" \
    --num_clients 10 \
    --partition iid \
    --num_rounds 100 \
    --local_epochs 5 \
    --local_lr 0.01 \
    --run_exact_shapley \
    --use_second_order \
    --dataset cifar10 \
    --seed 42
```

- `--run_exact_shapley`: 정확한 Shapley 값을 계산하여 In-run 결과와 비교합니다
- `--use_second_order`: 2차 Hessian 보정항을 포함합니다 (더 정확하지만 느림)

### 2. Non-IID 실험 (Dirichlet 분할)

```bash
python main.py \
    --exp_name "noniid_test" \
    --num_clients 10 \
    --partition dirichlet \
    --dirichlet_alpha 0.1 \
    --num_rounds 100 \
    --dataset cifar10
```

- `--dirichlet_alpha`: 작을수록 더 비균등한(non-IID) 분포
  - `0.1`: 매우 비균등 (각 클라이언트가 1-2개 클래스만 보유)
  - `0.5`: 적당히 비균등
  - `1.0`: 약간 비균등
  - `100.0`: 거의 IID와 동일

### 3. 노이즈 클라이언트 탐지

```bash
python main.py \
    --exp_name "noisy_detection" \
    --num_clients 10 \
    --noisy_clients 0 1 2 \
    --noise_type label_flip \
    --run_exact_shapley \
    --num_rounds 100 \
    --dataset cifar10
```

- `--noisy_clients 0 1 2`: 클라이언트 0, 1, 2에 노이즈 주입
- `--noise_type label_flip`: 라벨 뒤집기 공격 (label → num_classes - 1 - label)
- `--noise_type random_update`: 랜덤 노이즈 업데이트 전송

**기대 결과**: 노이즈 클라이언트는 음수(또는 매우 낮은) Shapley 값을 가지며, AUROC가 높을수록 탐지 성능이 좋습니다.

### 4. 데이터 크기 비균등 (Quantity Skew)

```bash
python main.py \
    --exp_name "quantity_skew" \
    --num_clients 10 \
    --partition iid \
    --quantity_skew \
    --quantity_beta 0.3 \
    --num_rounds 100
```

- `--quantity_skew`: 클라이언트별 데이터 양을 비균등하게 설정
- `--quantity_beta`: 작을수록 더 심한 크기 편향 (기본값: 0.5)

### 5. 중앙집중 학습 베이스라인 비교

```bash
python main.py \
    --exp_name "centralized_comparison" \
    --num_clients 10 \
    --run_centralized \
    --run_exact_shapley \
    --num_rounds 100
```

- `--run_centralized`: 모든 데이터를 합쳐 중앙집중으로 학습하며, 샘플 단위로 기여도를 클라이언트에 귀속시킵니다

### 6. GPU/CPU 선택

```bash
# GPU 사용 (기본값, CUDA가 없으면 자동으로 CPU 폴백)
python main.py --device cuda ...

# CPU 강제 사용
python main.py --device cpu ...
```

### 7. 로깅 설정

```bash
# Weights & Biases 로깅 활성화
python main.py \
    --use_wandb \
    --wandb_project "my_fl_project" \
    --wandb_entity "my_team" \
    ...

# TensorBoard 로깅 활성화
python main.py --use_tensorboard ...
# 확인: tensorboard --logdir outputs/tensorboard/
```

---

## 모듈별 상세 설명

### `main.py` — 실험 진입점

전체 실험 파이프라인을 관리합니다. 실행 순서:

1. **인자 파싱 및 시드 설정** (`config.get_args()`, `set_seed()`)
2. **데이터셋 로드** (`load_dataset()`) — CIFAR-10/100, Tiny ImageNet
3. **검증 셋 생성** — 학습 데이터의 마지막 `num_val_samples`개를 검증 데이터로 분리
4. **데이터 파티셔닝** (`partition_data()`) — 클라이언트별 데이터 인덱스 할당
5. **모델/서버/클라이언트 생성** — ResNet-18, Server, Client 인스턴스 초기화
6. **Shapley 계산기 생성** — InRunDataShapley (필수) + ExactShapley/MonteCarloShapley (선택)
7. **FL 학습 실행** (`FLTrainer.train()`) — 메인 학습 루프
8. **Ground-truth Shapley 계산** (선택) — Exact 또는 MC 방식
9. **중앙집중 학습 베이스라인** (선택)
10. **Fidelity 평가** — In-run vs Ground-truth 상관관계
11. **노이즈 클라이언트 탐지** — AUROC, Precision@k
12. **시각화 저장** — 바 차트, 산점도, 히트맵
13. **결과 파일 저장** — JSON, CSV

### `config.py` — 인자 정의

모든 실험 하이퍼파라미터를 `argparse`로 정의합니다. 주요 카테고리:

- **실험 설정**: `exp_name`, `seed`, `device`, `output_dir`
- **데이터**: `dataset`, `data_dir`
- **모델**: `model` (현재 `resnet18`만 지원)
- **FL 설정**: `num_clients`, `clients_per_round`, `num_rounds`, `local_epochs`, `local_lr`, `local_batch_size`, `aggregation`
- **파티셔닝**: `partition`, `dirichlet_alpha`, `quantity_skew`, `quantity_beta`
- **Shapley**: `use_second_order`, `num_val_samples`, `mc_permutations`
- **평가**: `eval_every`, `noisy_clients`, `noise_type`, `noise_ratio`, `run_centralized`, `run_exact_shapley`
- **로깅**: `use_wandb`, `wandb_project`, `use_tensorboard`, `log_every`

### `fl/server.py` — FL 서버

글로벌 모델을 관리하며 다음 기능을 제공합니다:

- **`aggregate()`**: FedAvg 집계 — 클라이언트 업데이트의 가중 평균으로 글로벌 모델 업데이트
  $$w_{t+1} = w_t + \sum_c \left( \text{weight}_c \times \Delta w_c \right)$$
- **`compute_validation_gradient()`**: 검증 데이터에 대한 gradient 계산 → 1D 텐서 반환
- **`compute_validation_hessian_vector_product(v)`**: Hessian-vector product (Hv) 계산. 이중 역전파(double backprop)를 사용하여 전체 Hessian을 물리화하지 않고 계산합니다.
- **`evaluate()`**: 검증/테스트 셋에서 loss와 accuracy 평가
- **`get_model_state()`**: 글로벌 모델 state_dict의 딥카피 반환
- **`get_trainable_param_names()`**: `requires_grad=True`인 파라미터 이름 목록 반환 (BatchNorm running stats 제외)

### `fl/client.py` — FL 클라이언트

로컬 SGD 학습을 수행합니다:

- **`local_train(global_model_state)`**: 글로벌 모델 가중치를 받아 로컬 학습 후 $\Delta w_c = w_{\text{local}} - w_{\text{global}}$ 반환
  - SGD (momentum=0.9, weight_decay=1e-4) 사용
  - 노이즈 클라이언트: `label_flip` → 라벨을 `(num_classes - 1 - label)`로 뒤집음
  - 노이즈 클라이언트: `random_update` → 학습 없이 랜덤 노이즈 업데이트 반환
- **`get_data_size()`**: 이 클라이언트가 보유한 학습 데이터 수 반환

### `fl/trainer.py` — FL 학습 루프

매 라운드마다 다음을 수행합니다:

1. `clients_per_round`만큼 클라이언트를 무작위 샘플링
2. 글로벌 모델 state를 각 클라이언트에 전달
3. 각 클라이언트의 로컬 업데이트 $\Delta w_c$ 수집
4. 업데이트를 1D 텐서로 flatten (trainable params만)
5. `InRunDataShapley.compute_round_shapley()` 호출
6. FedAvg 집계로 글로벌 모델 업데이트
7. 주기적으로 검증/테스트 평가 및 로깅

### `shapley/in_run_shapley.py` — In-Run Shapley (메인 방법)

가장 핵심적인 모듈입니다. 학습 중 추가 비용 없이 Shapley Value를 계산합니다.

**`compute_round_shapley()`** 동작 과정:
1. `server.compute_validation_gradient()` 호출 → $\nabla \ell(w_t, D_{\text{val}})$ (1D 텐서)
2. **1차항**: 각 클라이언트에 대해 $\phi_c^{(1)} = -\eta \cdot \langle \nabla \ell, \Delta w_c \rangle$
3. **2차항** (선택): $\phi_c^{(2)} = \frac{\eta^2}{2} \cdot \Delta w_c^\top H \sum_{c'} \Delta w_{c'}$
   - $H \cdot \sum \Delta w_c$는 `server.compute_validation_hessian_vector_product()`로 계산
4. $\phi_c^{(t)} = \phi_c^{(1)} + \phi_c^{(2)}$ 저장 및 누적

**시간 복잡도**: 1차항 $O(P \times C)$, 2차항 $O(P \times C + \text{backward\_pass})$ ($P$ = 파라미터 수, $C$ = 클라이언트 수)

### `shapley/exact_shapley.py` — Exact Shapley

모든 $2^n$ 부분집합을 열거하여 정확한 Shapley 값을 계산합니다.

- **제한**: $n \leq 10$ (assertion으로 강제)
- **유틸리티 함수**: $U(S) = \ell(w_t, D_{\text{val}}) - \ell\!\left(w_t + \eta \sum_{c \in S} \Delta w_c,\; D_{\text{val}}\right)$
- **Shapley 공식**: $\displaystyle \phi_c = \frac{1}{n!} \sum_{S \subseteq [n] \setminus \{c\}} |S|!\,(n-1-|S|)!\;\left[U(S \cup \{c\}) - U(S)\right]$
- 비트마스크를 사용하여 부분집합을 효율적으로 열거합니다

### `shapley/mc_shapley.py` — Monte Carlo Shapley

랜덤 순열 샘플링으로 Shapley 값을 근사합니다.

- **사용 시점**: n > 10일 때 (Exact 불가능)
- **알고리즘**: T개의 랜덤 순열에 대해 각 클라이언트의 한계 기여(marginal contribution)를 평균
- **수렴**: $T \to \infty$에서 Exact Shapley로 수렴
- `--mc_permutations`으로 순열 수 조절 (기본값: 1000)

### `data/datasets.py` — 데이터셋 로딩

| 데이터셋 | 이미지 크기 | 클래스 수 | 학습 샘플 | 테스트 샘플 |
|----------|-----------|----------|----------|-----------|
| CIFAR-10 | 32x32 | 10 | 50,000 | 10,000 |
| CIFAR-100 | 32x32 | 100 | 50,000 | 10,000 |
| Tiny ImageNet | 64x64 | 200 | 100,000 | 10,000 |

**데이터 증강**:
- 학습: RandomCrop + RandomHorizontalFlip + Normalize
- 테스트: Normalize만 적용

CIFAR는 `torchvision`으로 자동 다운로드, Tiny ImageNet은 수동 다운로드 후 ImageFolder 호환 구조로 변환합니다.

### `data/partition.py` — 데이터 파티셔닝

**IID 파티셔닝**: 전체 데이터를 랜덤으로 섞은 후 균등 분배

**Dirichlet non-IID 파티셔닝**:
- 각 클래스별로 `Dir(α)` 분포에서 클라이언트 비율을 샘플링
- α가 작을수록 (예: 0.1) 각 클라이언트가 소수 클래스에 편중
- α가 클수록 (예: 100) IID에 가까워짐

**Quantity Skew**: 클라이언트별 데이터 양도 `Dir(β)`로 결정. `β`가 작을수록 극단적인 크기 편향

### `models/resnet.py` — ResNet-18 모델

표준 torchvision ResNet-18을 데이터셋에 맞게 수정합니다:

- **CIFAR (32x32)**: 첫 번째 conv를 7x7(stride=2) → 3x3(stride=1)로 변경, MaxPool 제거
  - 이유: 32x32 이미지에 7x7 conv + stride=2를 적용하면 공간 해상도가 너무 줄어듦
- **Tiny ImageNet (64x64)**: MaxPool만 Identity로 교체 (과도한 다운샘플링 방지)

### `eval/fidelity.py` — Fidelity 평가

In-run Shapley와 Ground-truth Shapley 간의 상관관계를 측정합니다:

| 메트릭 | 설명 | 좋은 값 |
|--------|------|---------|
| **Spearman $\rho$** | 순위 상관계수 (가장 중요) | $\approx 1.0$ |
| **Pearson $r$** | 선형 상관계수 | $\approx 1.0$ |
| **RMSE** | 절대 오차 | $\approx 0$ |
| **MAE** | 평균 절대 오차 | $\approx 0$ |

Spearman이 가장 중요한 이유: Shapley 값은 보통 **순위**(ranking)에 사용되므로, 절대값보다 순서가 맞는지가 더 중요합니다.

### `eval/noisy_client.py` — 노이즈 클라이언트 탐지

Shapley 값을 사용하여 노이즈 클라이언트를 이진 분류합니다:

- **AUROC**: `-shapley_value`를 "노이즈 스코어"로 사용하여 ROC-AUC 계산
  - 1.0 = 완벽한 탐지 (모든 노이즈 클라이언트가 clean 클라이언트보다 낮은 Shapley)
  - 0.5 = 랜덤 (Shapley가 노이즈에 대한 정보 없음)
- **Precision@k**: 하위 k개 클라이언트 중 실제 노이즈 비율 (k = 노이즈 클라이언트 수)
- **Separation Ratio**: (clean 평균 - noisy 평균) / 전체 표준편차

### `eval/client_removal.py` — 클라이언트 제거 실험

Shapley 값의 유효성을 검증합니다:

1. **High-to-low 제거**: Shapley가 높은 클라이언트부터 제거 → 정확도 급락해야 함
2. **Low-to-high 제거**: Shapley가 낮은 클라이언트부터 제거 → 정확도 유지해야 함
3. **Random 제거**: 중간 수준의 정확도 감소 (베이스라인)

좋은 Shapley 지표라면: `acc(remove_high) << acc(remove_random) < acc(remove_low)`

### `centralized/centralized_trainer.py` — 중앙집중 학습

모든 클라이언트 데이터를 합쳐서 단일 모델을 학습하며, 각 샘플의 gradient 기여도를 원래 클라이언트에 귀속시킵니다:

$$\phi_c^{(\text{iter})} = \sum_{z_i \in D_c} -\eta \cdot \langle \nabla \ell(w, z_{\text{val}}),\; \nabla \ell(w, z_i) \rangle$$

`IndexedDataset` 래퍼를 사용하여 DataLoader에서 글로벌 인덱스를 추적합니다.

### `utils/timer.py` — 비용 추적 (CostTracker)

학습의 각 단계(phase)별 벽시계 시간과 FLOPs를 추적합니다:

- **`start(name)` / `stop(name)`**: 특정 phase의 시간 측정 (CUDA synchronize 포함)
- **`track(name)`**: context manager 방식 (`with cost_tracker.track("phase"):`)
- **`estimate_model_flops(model, input_shape)`**: PyTorch `FlopCounterMode`로 forward pass당 FLOPs 추정. PyTorch < 2.1 fallback: `2 × total_params`
- **`add_forward_passes(name, count)`**: phase별 forward pass 횟수 기록
- **`get_summary()`**: 전체 비용 요약 dict 반환 (results.json에 포함됨)
- **`print_summary()`**: 포맷된 비용 보고서 콘솔 출력

**추적되는 phase들**: `local_training`, `shapley_computation`, `aggregation`, `evaluation`, `fl_training_total`, `gt_shapley`, `centralized_training`

**FLOPs 추정 공식**: `총 FLOPs = forward_passes × flops_per_forward × 3` (backward ≈ 2× forward)

### `utils/logger.py` — 실험 로깅

세 가지 백엔드를 동시에 지원합니다:

1. **디스크 로깅** (항상 활성): JSON, CSV 파일 저장
2. **Weights & Biases** (`--use_wandb`): 실시간 메트릭 대시보드
3. **TensorBoard** (`--use_tensorboard`): 텐서보드 대시보드

추가 기능:
- **라운드별 타이밍 CSV 저장**: `{tag}_timing_rounds.csv` — 각 라운드의 `local_training`, `shapley_computation`, `aggregation`, `evaluation` 시간
- 실험 태그 자동 생성 예시: `cifar10_c5_k5_iid_r50_e5_lr0.01_noisy4_s42`

### `utils/visualize.py` — 시각화

| 시각화 | 함수 | 파일명 |
|--------|------|--------|
| Shapley 바 차트 | `plot_shapley_bar()` | `{tag}_shapley_bar.png` |
| Fidelity 산점도 | `plot_fidelity_scatter()` | `{tag}_fidelity_scatter.png` |
| 파티션 히트맵 | `plot_partition_heatmap()` | `{tag}_partition_heatmap.png` |
| 클라이언트 제거 곡선 | `plot_client_removal()` | `client_removal.png` |

---

## 비용 추적 (시간 및 FLOPs)

학습과 기여도 계산의 비용을 정량적으로 파악하기 위해, 모든 실험에서 자동으로 벽시계 시간과 FLOPs를 기록합니다.

### 자동 추적 항목

| Phase | 설명 | 비고 |
|-------|------|------|
| `local_training` | 클라이언트 로컬 SGD 학습 시간 | 전체 라운드 합산 |
| `shapley_computation` | In-Run Shapley 계산 시간 | val gradient + dot product |
| `aggregation` | FedAvg 집계 시간 | 일반적으로 매우 짧음 |
| `evaluation` | 검증/테스트 평가 시간 | `--eval_every` 주기 |
| `fl_training_total` | FL 학습 전체 시간 | 위 4개의 상위 합계 |
| `gt_shapley` | Ground-truth Shapley 계산 시간 | `--run_exact_shapley` 시 |
| `centralized_training` | 중앙집중 학습 시간 | `--run_centralized` 시 |

### 출력 예시

```
============================================================
  Cost Summary (Wall-Clock Time & Estimated FLOPs)
============================================================
  [Time]
    fl_training_total             :    3612.0s total
    local_training                :    3509.0s total (70.2s avg × 50 calls)
    shapley_computation           :      10.5s total (0.21s avg × 50 calls)
    gt_shapley                    :      77.6s total
    centralized_training          :   12071.0s total

  [FLOPs]
    Per forward pass: 1.11 GFLOPs
    local_training                :  193750 fwd passes -> 645.68 TFLOPs
    shapley_computation           :     400 fwd passes -> 1.33 TFLOPs
============================================================
```

### Shapley Overhead

Shapley 계산의 오버헤드는 전체 학습 시간 대비 **0.1%~1.5%** 수준으로, 실질적으로 무시할 수 있습니다:

| 설정 | Shapley Overhead |
|------|-----------------|
| 1st order, 50 rounds | ~0.3% |
| 2nd order, 50 rounds | ~1.0% |
| 250 rounds | ~1.5% |

---

## 실험 스크립트

### 메인 실험 스위트 (3개 실험, 39 runs)

```bash
cd fed_shapley

# 전체 실험 실행 (39 runs, GPU 기준 ~61시간)
bash scripts/run_experiments.sh

# 개별 실험만 실행
bash scripts/run_experiments.sh --exp 1    # Exp1: Non-FL vs FL (6 runs)
bash scripts/run_experiments.sh --exp 2    # Exp2: Communication trade-off (15 runs)
bash scripts/run_experiments.sh --exp 3    # Exp3: Non-IID sweep (18 runs)

# 명령어만 확인 (실행하지 않음)
bash scripts/run_experiments.sh --dry-run
```

### 결과 분석

```bash
# 전체 실험 결과 집계 및 요약 테이블 출력
python scripts/analyze_results.py --results_root ../results
# → results/analysis/exp{1,2,3}_summary.csv 생성
```

### 기타 실험 스크립트

```bash
# 기본 IID 실험 (10 클라이언트, 100 라운드, ~30분 GPU)
bash scripts/run_basic.sh

# Non-IID 알파 스윕 (alpha = 0.1, 0.5, 1.0, 5.0, 100.0)
bash scripts/run_noniid.sh

# 로컬 에폭 스윕 (K=1,2,5,10,20,50, 총 계산량 고정)
bash scripts/run_k_sweep.sh
```

### 그리드 서치

YAML 설정 파일로 다양한 하이퍼파라미터 조합을 자동 실행합니다.

```bash
# 알파 스윕 (15 실험 = 5 alpha × 3 seeds)
python scripts/run_grid_search.py --config scripts/grid_configs/alpha_sweep.yaml

# 노이즈 스윕
python scripts/run_grid_search.py --config scripts/grid_configs/noise_sweep.yaml

# 드라이런 (실행 없이 명령어만 확인)
python scripts/run_grid_search.py --config scripts/grid_configs/k_sweep.yaml --dry_run

# 최대 실행 수 제한 (테스트용)
python scripts/run_grid_search.py --config scripts/grid_configs/alpha_sweep.yaml --max_runs 3
```

---

## 실험 결과 요약

CIFAR-10, 5 클라이언트 환경에서 수행된 3가지 실험의 결과입니다. 모든 실험에서 클라이언트 4에 label_flip 노이즈를 주입하고, Exact Shapley를 ground-truth로 사용했습니다. 각 설정은 3개의 seed (42, 123, 456)로 반복 실행되었습니다.

### 실험 1: Non-FL vs FL Shapley 정확도 비교

FL In-Run Shapley와 Centralized(비-FL) 방식의 기여도 평가 정확도를 Exact Shapley 대비 비교합니다.

| 방법 | Spearman $\rho$ | Pearson $r$ | AUROC | FL 시간 | 비고 |
|------|----------------|-------------|-------|---------|------|
| FL In-Run (1st order) | $0.60 \pm 0.36$ | $0.91 \pm 0.08$ | $1.0$ | ~3,606s | 기본 방법 |
| FL In-Run (2nd order) | $0.53 \pm 0.39$ | $0.84 \pm 0.08$ | $1.0$ | ~3,632s | Hessian 보정 |
| Centralized | $-0.30 \pm 0.28$ | $-0.31 \pm 0.26$ | - | ~12,012s | 비-FL baseline |

**핵심 발견**:
- FL In-Run이 Centralized보다 ranking 정확도가 높음 ($\rho$ 0.6 vs -0.3)
- FL이 Centralized보다 **3.3배 빠름** (~1시간 vs ~3.3시간)
- 2nd order Hessian 보정은 이 설정에서 유의한 개선을 보이지 않음
- **노이즈 클라이언트 탐지는 모든 경우 AUROC 1.0** (완벽한 탐지)

### 실험 2: Communication Cost vs Accuracy Trade-off

총 computation budget를 $\text{rounds} \times \text{epochs} = 250$으로 고정하고, round를 촘촘하게 나눌수록 Shapley 정확도가 어떻게 변하는지 측정합니다.

| Rounds | Epochs | Budget | Spearman $\rho$ | AUROC | Shapley OH% | 총 시간 |
|--------|--------|--------|----------------|-------|-------------|---------|
| 10 | 25 | 250 | $0.33 \pm 0.05$ | $1.0$ | 0.06% | ~3,541s |
| **25** | **10** | **250** | $\mathbf{0.63 \pm 0.09}$ | $\mathbf{1.0}$ | **0.15%** | **~3,543s** |
| 50 | 5 | 250 | $0.60 \pm 0.36$ | $1.0$ | 0.30% | ~3,609s |
| **125** | **2** | **250** | $\mathbf{0.63 \pm 0.25}$ | $\mathbf{1.0}$ | **0.76%** | **~3,767s** |
| 250 | 1 | 250 | $0.57 \pm 0.19$ | $1.0$ | 1.50% | ~4,045s |

**핵심 발견**:
- **Rounds 25~125이 최적 trade-off** ($\rho \approx 0.63$)
- 너무 적은 rounds (10): Shapley 갱신 빈도 부족으로 정확도 하락
- 너무 많은 rounds (250): 짧은 local training으로 업데이트 신호 약화
- Shapley overhead는 최대 1.5%로 실질적으로 무시 가능
- 총 시간은 모든 설정에서 유사 (동일 computation budget)

### 실험 3: Non-IID 수준 vs Shapley 정확도

Dirichlet $\alpha$를 변화시켜 데이터 이질성이 기여도 평가 정확도에 미치는 영향을 측정합니다.

| 파티션 | $\alpha$ | Non-IID 수준 | Spearman $\rho$ | AUROC | Test Acc |
|--------|---------|------------|----------------|-------|----------|
| IID | - | 기준 (균일) | $0.60 \pm 0.36$ | $1.0$ | $0.219$ |
| Dirichlet | 10.0 | 약함 | $0.50 \pm 0.28$ | $1.0$ | $0.218$ |
| Dirichlet | 1.0 | 중간 | $0.33 \pm 0.40$ | $1.0$ | $0.210$ |
| Dirichlet | 0.5 | 강함 | $0.57 \pm 0.12$ | $1.0$ | $0.244$ |
| Dirichlet | **0.1** | **매우 강함** | $\mathbf{-0.23 \pm 0.17}$ | $1.0$ | $0.192$ |
| Dirichlet | 0.01 | 극단적 | $0.47 \pm 0.26$ | $0.92$ | $0.135$ |

**핵심 발견**:
- IID ~ 약한 non-IID ($\alpha \geq 0.5$)에서 양호한 ranking 상관 ($\rho \approx 0.5\text{--}0.6$)
- **매우 강한 non-IID ($\alpha = 0.1$)에서 Shapley ranking이 역전** ($\rho$ 음수) — gradient 방향의 이질성이 dot-product 기반 추정을 교란
- 극단적 non-IID ($\alpha = 0.01$)에서 AUROC도 0.92로 하락 — 노이즈 탐지 성능 저하 시작

---

**YAML 설정 파일 형식** (`grid_configs/alpha_sweep.yaml` 예시):

```yaml
base_args:
  dataset: cifar10
  num_clients: 10
  num_rounds: 100
  local_epochs: 5
  local_lr: 0.01
  partition: dirichlet
  run_exact_shapley: true
  use_second_order: true

grid:
  dirichlet_alpha: [0.1, 0.5, 1.0, 5.0, 100.0]
  seed: [42, 123, 456]
```

`base_args`는 모든 실험에 공통 적용되고, `grid`의 값들의 모든 조합(Cartesian product)이 실행됩니다. 결과는 CSV 요약 파일로 저장됩니다.

---

## 출력 결과 해석

### 출력 파일 구조

모든 결과는 `--output_dir` 디렉토리에 실험 태그 접두사와 함께 저장됩니다:

```
{output_dir}/
├── {tag}_results.json           # 전체 실험 결과 (config, metrics, cost_summary 포함)
├── {tag}_shapley.csv            # 누적 Shapley 값
├── {tag}_shapley_rounds.csv     # 라운드별 Shapley 값
├── {tag}_timing_rounds.csv      # 라운드별 시간 측정 (local_training, shapley, ...)
└── figures/
    ├── {tag}_shapley_bar.png
    ├── {tag}_fidelity_scatter.png
    └── {tag}_partition_heatmap.png
```

태그 예시: `cifar10_c5_k5_iid_r50_e5_lr0.01_noisy4_s42`

### Shapley 값 해석

| Shapley 값 | 의미 |
|------------|------|
| $\phi_c > 0$ (양수) | 클라이언트 $c$의 업데이트가 검증 성능을 **향상**시킴 (유익) |
| $\phi_c < 0$ (음수) | 클라이언트 $c$의 업데이트가 검증 성능을 **해침** (유해/노이즈) |
| $\phi_c \approx 0$ | 클라이언트 $c$의 영향이 미미함 |

### results.json 구조

```json
{
  "experiment_tag": "cifar10_c5_k5_iid_r50_e5_lr0.01_noisy4_s42",
  "config": { ... },                // 전체 실험 설정
  "round_logs": [                   // 라운드별 메트릭
    {"round": 1, "val_loss": 1.8, "val_acc": 0.35, "test_loss": 1.9, "test_acc": 0.33},
    ...
  ],
  "final_summary": {                // 최종 요약 메트릭
    "final_val_acc": 0.87,
    "final_test_acc": 0.86,
    "fidelity_spearman_r": 0.90,
    "fidelity_pearson_r": 0.97,
    "detection/auroc": 1.0,
    "detection/precision_at_k": 1.0,
    "time/fl_training_total_total_sec": 3612.0,
    "time/local_training_total_sec": 3509.0,
    "time/shapley_computation_total_sec": 10.5,
    "flops/local_training_estimated_total_flops": 645680000000,
    ...
  }
}
```

---

## 전체 인자(Arguments) 목록

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| **실험** | | | |
| `--exp_name` | str | `"exp"` | 실험 이름 (출력 파일 명명에 사용) |
| `--seed` | int | `42` | 재현성을 위한 랜덤 시드 |
| `--device` | str | `"cuda"` | `"cuda"` 또는 `"cpu"` |
| `--output_dir` | str | `"./outputs"` | 결과 저장 디렉토리 |
| **데이터** | | | |
| `--dataset` | str | `"cifar10"` | `cifar10`, `cifar100`, `tinyimagenet` 중 선택 |
| `--data_dir` | str | `"./data_cache"` | 데이터셋 캐시 디렉토리 |
| **모델** | | | |
| `--model` | str | `"resnet18"` | 모델 아키텍처 (현재 `resnet18`만 지원) |
| **FL 설정** | | | |
| `--num_clients` | int | `10` | 전체 FL 클라이언트 수 |
| `--clients_per_round` | int | `None` | 라운드당 참여 클라이언트 수 (None = 전체) |
| `--num_rounds` | int | `100` | FL 통신 라운드 수 |
| `--local_epochs` | int | `5` | 클라이언트 로컬 SGD 에폭 수 |
| `--local_lr` | float | `0.01` | 클라이언트 SGD 학습률 |
| `--local_batch_size` | int | `64` | 로컬 미니배치 크기 |
| `--aggregation` | str | `"fedavg"` | 서버 집계 방식 |
| **파티셔닝** | | | |
| `--partition` | str | `"iid"` | `iid` 또는 `dirichlet` |
| `--dirichlet_alpha` | float | `0.5` | Dirichlet 집중 파라미터 (작을수록 non-IID) |
| `--quantity_skew` | flag | `False` | 클라이언트별 데이터 양 비균등 적용 |
| `--quantity_beta` | float | `0.5` | 양적 편향의 Dirichlet 집중 파라미터 |
| **Shapley** | | | |
| `--use_second_order` | flag | `False` | 2차 Hessian 보정항 포함 |
| `--num_val_samples` | int | `500` | 학습 데이터에서 분리할 검증 샘플 수 |
| `--mc_permutations` | int | `1000` | Monte Carlo Shapley의 순열 수 |
| **평가** | | | |
| `--eval_every` | int | `1` | 평가 주기 (라운드 단위) |
| `--noisy_clients` | int[] | `[]` | 노이즈 주입할 클라이언트 ID 목록 |
| `--noise_type` | str | `"label_flip"` | `label_flip` 또는 `random_update` |
| `--noise_ratio` | float | `0.5` | 노이즈 비율 (현재 정보 제공용) |
| `--run_centralized` | flag | `False` | 중앙집중 학습 베이스라인 실행 |
| `--run_exact_shapley` | flag | `False` | Exact/MC Shapley 비교 실행 |
| **로깅** | | | |
| `--use_wandb` | flag | `False` | W&B 로깅 활성화 |
| `--wandb_project` | str | `"fed_shapley"` | W&B 프로젝트명 |
| `--wandb_run_name` | str | `None` | W&B 실행명 (None = 자동 태그) |
| `--wandb_entity` | str | `None` | W&B 팀/사용자명 |
| `--use_tensorboard` | flag | `False` | TensorBoard 로깅 활성화 |
| `--log_every` | int | `1` | 로깅 주기 (라운드 단위) |

---

## 수학적 배경

### Shapley Value 정의

$n$명의 플레이어(클라이언트)가 있을 때, 플레이어 $c$의 Shapley 값은:

$$\phi_c = \frac{1}{n!} \sum_{\pi \in \Pi} \left[ U(S_\pi^c \cup \{c\}) - U(S_\pi^c) \right]$$

여기서:
- $\Pi$: 모든 $n!$ 순열의 집합
- $S_\pi^c$: 순열 $\pi$에서 $c$ 앞에 오는 플레이어들의 집합
- $U(S)$: 플레이어 집합 $S$의 유틸리티 (여기서는 검증 손실 감소량)

### FL에서의 유틸리티 함수

$$U(S) = \ell(w_t, D_{\text{val}}) - \ell\!\left(w_t + \eta \sum_{c \in S} \Delta w_c,\; D_{\text{val}}\right)$$

즉, 부분집합 $S$의 클라이언트 업데이트를 적용했을 때 검증 손실이 얼마나 감소하는지를 측정합니다.

### In-Run 근사 (1차)

Taylor 전개를 적용하면:

$$\ell(w + \delta) \approx \ell(w) + \nabla \ell(w)^\top \delta + \frac{1}{2} \delta^\top H \delta + \cdots$$

1차항만 사용하면:

$$U(S) \approx -\eta \sum_{c \in S} \langle \nabla \ell(w_t, D_{\text{val}}),\; \Delta w_c \rangle$$

이것의 Shapley 분해는 (선형성에 의해) 단순히:

$$\phi_c^{(t)} = -\eta \cdot \langle \nabla \ell(w_t, D_{\text{val}}),\; \Delta w_c^{(t)} \rangle$$

**핵심**: 1차 근사 하에서 Shapley 값은 단순한 gradient dot-product로 분해되므로, 지수적 부분집합 열거 없이 $O(P \times C)$에 계산할 수 있습니다.

### 2차 보정

2차항까지 포함하면:

$$\phi_c^{(2)} = \frac{\eta^2}{2} \cdot \Delta w_c^\top \, H(D_{\text{val}}) \sum_{c'} \Delta w_{c'}$$

이는 Hessian-vector product로 계산합니다 (전체 Hessian 물리화 불필요).

### 누적 Shapley

여러 라운드에 걸친 총 기여도:

$$\phi_c = \sum_{t:\, c \in C_t} \phi_c^{(t)}$$

여기서 $C_t$는 라운드 $t$에 참여한 클라이언트 집합입니다.

---

## 인용

```bibtex
@inproceedings{inrun2025,
  title={Data Shapley in One Training Run},
  booktitle={ICLR 2025},
  year={2025}
}
```
