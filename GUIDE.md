# 실행 가이드: Joint Rotation 기반 Neural ODE & Hamiltonian Neural ODE

## 전제 조건

### 환경 설정

```bash
conda create -n joint_ode python=3.9
conda activate joint_ode

# PyTorch (CUDA 11.8 기준)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 일반 패키지
pip install -r requirements.txt

pip install git+https://github.com/facebookresearch/pytorch3d.git
(or https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md 참고)
# conda install pytorch3d -c pytorch3d -c pytorch -c conda-forge

# 3DGS 렌더러 (RigGS repo 루트에서 실행)
# diff-gaussian-rasterization: 실제 Gaussian rasterizer (렌더링 필수)
pip install ./submodules/diff-gaussian-rasterization --no-build-isolation
pip install ./submodules/simple-knn --no-build-isolation
```

> **렌더링 백엔드 우선순위**
> 1. `diff-gaussian-rasterization` (3DGS/RigGS 동일 CUDA 래스터라이저, 권장)
> 2. `gsplat` (pure-pip 대체재, diff-gaussian-rasterization 없을 때 자동 사용)

### 입력 데이터

RigGS가 출력한 `joint_trajectory.npz` 파일을 입력으로 사용합니다.

```
예시 경로: /home/airlab/RigGS/output/standup/standup_node/train/ours_100000/joint_trajectory.npz
형식: numpy npz — joint_rotation [T, N_j, 4] (quaternion), timestamps [T], parent_indices [N_j]
```

---

## 1. Neural ODE 학습 (ODE-GS 구조 기반)

### 구조 설명

```
관측 구간 (80%) ──► Transformer 인코더 ──► z0 (latent)
                                              │
                                         LatentODE dz/dt
                                              │
                                         z_traj [T_ext, 64]
                                              │
                                         디코더 ──► θ_pred [T_ext, N_j, 4]

외삽 구간 (20%) ◄── 예측
```

### 학습 실행

```bash
python -m joint_ham_ode.train \
    --config joint_ham_ode/configs/neural_ode.yaml \
    --theta_path /home/airlab/RigGS/output/standup/standup_node/train/ours_100000/joint_trajectory.npz \
    --output_dir output/standup/neural_ode \
    --device cuda
```

### 학습 중 출력 예시

```
Loaded trajectory: T=100, N_j=18, rot_dim=4
T=100  N_j=18  T_train=80  T_extrap=20
Model: neural_ode  params: 412,416
[    0/2000] loss=0.312541  recon=0.312021  kl=0.520134  lr=1.00e-03
[   50/2000] loss=0.198432  recon=0.198102  kl=0.330124  lr=9.97e-04
...
```

### 주요 하이퍼파라미터 (`configs/neural_ode.yaml`)

| 항목 | 기본값 | 설명 |
|---|---|---|
| `data.time_split` | 0.8 | 학습/외삽 비율 (8:2) |
| `model.latent_dim` | 64 | latent 공간 차원 |
| `model.d_model` | 128 | Transformer d_model |
| `model.ode_nhidden` | 256 | ODE MLP hidden dim |
| `training.epochs` | 2000 | 학습 epoch 수 |
| `training.lr` | 1e-3 | learning rate |
| `model.kl_beta` | 1e-3 | VAE KL loss 가중치 |

---

## 2. Hamiltonian ODE 학습

### 구조 설명

```
q0 = θ_train[0]  (초기 joint 각도)
p0 = dθ/dt|t=0  (초기 angular velocity, 중심 차분으로 추정)

상태: z = [q | p]  (2 × N_j × 4 차원)

dynamics:  dq/dt = p
           dp/dt = -∂V_ψ(q)/∂q       ← V_ψ: MLP (학습 대상)

에너지:    H(q, p) = ½||p||² + V_ψ(q) = 상수 (보존)
```

### 학습 실행

```bash
python -m joint_ham_ode.train \
    --config joint_ham_ode/configs/hamiltonian_ode.yaml \
    --theta_path /home/airlab/RigGS/output/standup/standup_node/train/ours_100000/joint_trajectory.npz \
    --output_dir output/standup/ham_ode \
    --device cuda
```

### 학습 중 출력 예시

```
Loaded trajectory: T=100, N_j=18, rot_dim=4
T=100  N_j=18  T_train=80  T_extrap=20
Model: hamiltonian_ode  params: 527,617
[    0/2000] loss=0.318241  recon=0.317893  energy=0.003481  H=0.2341  lr=1.00e-03
[   50/2000] loss=0.201234  recon=0.200912  energy=0.000322  H=0.2289  lr=9.97e-04
...
```

- `energy` 값이 점차 0에 가까워지면 에너지 보존 학습 중
- `H` 값이 안정되면 Hamiltonian이 수렴한 것

### 주요 하이퍼파라미터 (`configs/hamiltonian_ode.yaml`)

| 항목 | 기본값 | 설명 |
|---|---|---|
| `data.time_split` | 0.8 | 학습/외삽 비율 (8:2) |
| `model.hidden_dim` | 256 | V_ψ MLP hidden dim |
| `model.n_layers` | 3 | V_ψ MLP 레이어 수 |
| `training.lambda_energy` | 0.01 | 에너지 보존 loss 가중치 |
| `training.energy_subsample` | 5 | 에너지 loss 계산 간격 |

---

## 3. 평가

### 평가 실행

#### Joint MAE만 (렌더링 없음)

```bash
# Neural ODE
python -m joint_ham_ode.evaluate \
    --checkpoint output/standup/neural_ode/model_final.pt \
    --theta_path /home/airlab/RigGS/output/standup/standup_node/train/ours_100000/joint_trajectory.npz \
    --output_dir output/standup/neural_ode \
    --device cuda

# Hamiltonian ODE
python -m joint_ham_ode.evaluate \
    --checkpoint output/standup/ham_ode/model_final.pt \
    --theta_path /home/airlab/RigGS/output/standup/standup_node/train/ours_100000/joint_trajectory.npz \
    --output_dir output/standup/ham_ode \
    --device cuda
```

#### 렌더링 평가 포함 (PSNR / LPIPS / MP4)

`--riggs_model_path`와 `--dataset_path`를 추가하면 **joint_ode 환경만으로** (RigGS conda 환경 불필요)
실제 렌더링 후 이미지·영상까지 저장합니다.

| 인자 | 가리키는 경로 | 포함 내용 |
|---|---|---|
| `--riggs_model_path` | RigGS output 루트 | `point_cloud/`, `skeleton/`, `skeleton_tree.npz` |
| `--dataset_path` | D-NeRF 데이터셋 루트 | `transforms_train.json`, GT 이미지 |

```bash
TRAJ=/home/airlab/RigGS/output/standup/standup_node/train/ours_100000/joint_trajectory.npz
RIGGS=/home/airlab/RigGS/output/standup/standup_node
DATA=/home/airlab/RigGS/data/dy_syn_data/standup          # transforms_train.json + GT 이미지 위치

python -m joint_ham_ode.evaluate \
    --checkpoint output/standup/neural_ode/model_final.pt \
    --theta_path $TRAJ \
    --output_dir output/standup/neural_ode \
    --riggs_model_path $RIGGS \
    --dataset_path $DATA \
    --image_size 800 \
    --background 0.0 \
    --render_fps 10 \
    --device cuda
```

> `--dataset_path` 없이 실행하면 렌더링은 되지만 GT 이미지가 없어 PSNR/LPIPS는 계산되지 않습니다.

### 출력 결과

```
=== Joint MAE ===
  Interp:  2.3412 deg    ← 학습 구간 (0~80%)
  Extrap:  5.7823 deg    ← 외삽 구간 (80~100%)

=== Energy Conservation (extrap) ===   ← Hamiltonian만
  ΔH_mean: 0.001234
  ΔH_max:  0.003421

=== Rendering Evaluation ===
  [interp]  PSNR=28.41 dB  LPIPS=0.0823
  [extrap]  PSNR=24.17 dB  LPIPS=0.1241
```

### 저장되는 파일

```
output/standup/ham_ode/
├── model_final.pt                     # 최종 모델 체크포인트
├── history.json                       # epoch별 loss 기록
├── eval_results.json                  # 전체 평가 지표 (JSON)
├── mae_per_joint_interp.pt            # joint별 MAE (학습 구간)
├── mae_per_joint_extrap.pt            # joint별 MAE (외삽 구간)
├── H_extrap.pt                        # 에너지 곡선 (Hamiltonian만)
├── render_interp/
│   ├── pred/          pred_0000.png … # 예측 렌더링 이미지
│   ├── gt/            gt_0000.png …   # GT 이미지
│   ├── comparison/    cmp_0000.png …  # pred | gt 비교 이미지
│   ├── pred_interp.mp4                # 예측 렌더링 영상
│   └── comparison_interp.mp4         # pred | gt 비교 영상
└── render_extrap/
    ├── pred/          pred_0000.png …
    ├── gt/            gt_0000.png …
    ├── comparison/    cmp_0000.png …
    ├── pred_extrap.mp4                # ← 핵심: 외삽 구간 렌더링 영상
    └── comparison_extrap.mp4         # ← 핵심: 외삽 구간 비교 영상
```

---

## 4. 여러 시퀀스 비교 실험

D-NeRF 데이터셋 기준 우선순위: `standup` → `jumpingjacks` → `mutant`

```bash
for SEQ in standup jumpingjacks mutant; do
    TRAJ=/home/airlab/RigGS/output/${SEQ}/${SEQ}_node/train/ours_100000/joint_trajectory.npy

    python -m joint_ham_ode.train \
        --config joint_ham_ode/configs/neural_ode.yaml \
        --theta_path $TRAJ \
        --output_dir output/${SEQ}/neural_ode

    python -m joint_ham_ode.train \
        --config joint_ham_ode/configs/hamiltonian_ode.yaml \
        --theta_path $TRAJ \
        --output_dir output/${SEQ}/ham_ode

    python -m joint_ham_ode.evaluate \
        --checkpoint output/${SEQ}/neural_ode/model_final.pt \
        --theta_path $TRAJ \
        --output_dir output/${SEQ}/neural_ode

    python -m joint_ham_ode.evaluate \
        --checkpoint output/${SEQ}/ham_ode/model_final.pt \
        --theta_path $TRAJ \
        --output_dir output/${SEQ}/ham_ode
done
```

---

## 5. 결과 비교 테이블 예시

05.20 update
=======================================================
Metric                         Neural ODE    Ham ODE
=======================================================
Joint MAE Interp (deg)            29.4445     0.7187
Joint MAE Extrap (deg)             1.6333    39.5549
PSNR Interp (dB)                  19.5128    21.4503
PSNR Extrap (dB)                  18.6279    18.4794
LPIPS Interp                       0.1404     0.1019
LPIPS Extrap                       0.1782     0.1756
-------------------------------------------------------
ΔH_mean (Ham only)                        0.008344
ΔH_max  (Ham only)                        0.015235
=======================================================


**Phase 1 통과 기준**: Extrap MAE: Ham-ODE ≤ Neural ODE

---

## 6. 자주 발생하는 문제

### Q. `joint_trajectory.npy`의 shape이 [T, N_j, 3]이면?
axis-angle로 자동 인식해서 quaternion으로 변환합니다. 별도 처리 불필요.

### Q. CUDA out of memory 발생 시
`configs/hamiltonian_ode.yaml`에서 `model.use_adjoint: false`로 변경하거나
`training.energy_subsample`을 10으로 높이세요.

### Q. Hamiltonian loss가 수렴하지 않으면
`training.lambda_energy`를 0.001로 낮추거나
`training.lr`을 5e-4로 줄여보세요.

### Q. 에너지가 학습 초반에 발산하면
처음 200 epoch은 `lambda_energy=0`으로 시작하고 이후 점진적으로 올리는 것을 권장합니다.
