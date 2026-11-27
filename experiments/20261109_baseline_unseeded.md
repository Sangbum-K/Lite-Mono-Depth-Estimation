# Run: 20261109_baseline_unseeded

## Purpose

- Hypothesis: 공식 코드와 공식 하이퍼파라미터로 로컬 RTX 5070에서 학습하면 논문 Lite-Mono(640×192) 수치에 근접한다.
- Comparison target: 논문 Table 1 Lite-Mono 640×192 (AbsRel 0.107, SqRel 0.765, RMSE 4.561, RMSE log 0.183, δ₁ 0.886, δ₂ 0.963, δ₃ 0.983), 공개 checkpoint `pretrained/kitti_640x192`
- Decision this run will support: 로컬 학습 파이프라인을 이후 개선 실험의 baseline으로 써도 되는지

## Reproducibility

- Date: 2026-11-09 13:09 ~ 22:09 (KST)
- Code revision: 학습 시작 시 `Local` 브랜치 `fb80637` (evaluate_depth.py PNG 수정·export_gt_depth.py는 학습 후 추가, 학습 코드에는 영향 없음)
- Upstream Lite-Mono commit: `4874b35`
- Config or exact command: `scripts/train_baseline.sh 20261109_baseline_unseeded` → `train.py --data_path data/raw/kitti --png --split eigen_zhou --model lite-mono --height 192 --width 640 --batch_size 12 --num_epochs 30 --lr 0.0001 5e-6 31 0.0001 1e-5 31 --mypretrain pretrained/imagenet/lite-mono-pretrain.pth --num_workers 6`
- Evaluation: `scripts/eval_eigen.sh <weights_folder> <log>` (Eigen test 697장, median scaling, 80m cap, Garg crop, post-process 없음)
- Data split and hash (sha256):
  - `eigen_zhou/train_files.txt` 39,810: `00173a231642151c4693274e701ddd4cae7916520a757c7a94eae8e1173f7b0c`
  - `eigen_zhou/val_files.txt` 4,424: `64a868b8d2c66c3979a2d74c22ea2e9146850d5b9f5786ed8e5f92201ba1ad04`
  - `eigen/test_files.txt` 697: `e226d4ca16e0e6e9695435014347dcf973588531330536fd8a68d298ef091a6b`
- Image format: 원본 PNG (`--png`, 논문은 JPG)
- Initialization file and checksum: `pretrained/imagenet/lite-mono-pretrain.pth` sha256 `9877014dfaede372ca887dea354cb534788eb68b6ded13a9a1d39c715c04fc9f` (encoder 263개 중 233개 텐서 로드)
- Seed: 없음 (공식 코드와 같음)

## Training

- GPU / VRAM: RTX 5070 12GB (driver 580.178.04)
- Precision: FP32 (공식 코드 기본)
- Input resolution: 640×192
- Batch size: 12
- Epochs: 30 (epoch당 약 18분)
- Learning rate / scheduler: depth 1e-4 → 5e-6, pose 1e-4 → 1e-5, cosine warm restart 31 epoch (공식 README)
- Peak allocated / reserved memory: 측정 안 함. 학습 중 nvidia-smi 프로세스 사용량 약 8.4GB
- Wall time: 약 9시간 0분 (13:09 시작, `weights_29` 22:09 저장)
- Train loss: epoch 0 약 0.15 → epoch 29 약 0.07. NaN·에러 없음
- 기타: GPU 사용률 약 47%, 데이터 로딩 병목으로 추정 (num_workers 6, RAM 15GB 제약)
- Output path: `outputs/20261109_baseline_unseeded/` (models/weights_0~29, train/·val/ tensorboard, eval/weights_N.log)

## Results

Eigen test 697장. 논문 보고 방식대로 마지막 epoch(`weights_29`)를 대표 결과로 쓴다.

| Checkpoint | AbsRel | SqRel | RMSE | RMSE log | δ₁ | δ₂ | δ₃ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 논문 Lite-Mono 640×192 | 0.107 | 0.765 | 4.561 | 0.183 | 0.886 | 0.963 | 0.983 |
| 공개 checkpoint (PNG 입력으로 평가) | 0.107 | 0.766 | 4.560 | 0.183 | 0.886 | 0.963 | 0.983 |
| **이 run `weights_29` (최종)** | **0.107** | **0.821** | **4.608** | **0.184** | **0.887** | **0.962** | **0.982** |

- median scaling ratio: 이 run 32.491 ± 0.087, 공개 checkpoint 32.591 ± 0.088
- Parameters: 3.069M (encoder 2.842M + decoder 226.6K), FLOPs 5.032G (evaluate_depth.py thop 출력)
- Batch-1 latency: 측정 안 함
- Qualitative figure paths: `outputs/20261109_baseline_unseeded/analysis/weights_29/figs/` (worst 20 / best 5 / random 10, `overview_*.png`, `signed_*.png`). 생성: `analysis/baseline_failure_analysis.py`

### epoch별 추이 (참고용, test로 checkpoint를 고르지 않는다)

| epoch | AbsRel | SqRel | RMSE | RMSE log | δ₁ | δ₂ | δ₃ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.141 | 0.904 | 5.183 | 0.209 | 0.813 | 0.947 | 0.982 |
| 1 | 0.127 | 0.866 | 5.219 | 0.204 | 0.833 | 0.950 | 0.982 |
| 2 | 0.134 | 1.066 | 5.095 | 0.201 | 0.844 | 0.955 | 0.982 |
| 3 | 0.122 | 0.883 | 4.801 | 0.192 | 0.863 | 0.959 | 0.982 |
| 4 | 0.118 | 0.849 | 4.759 | 0.189 | 0.867 | 0.959 | 0.983 |
| 5 | 0.116 | 0.826 | 4.724 | 0.191 | 0.868 | 0.960 | 0.982 |
| 6 | 0.115 | 0.849 | 4.710 | 0.189 | 0.872 | 0.960 | 0.982 |
| 7 | 0.112 | 0.791 | 4.662 | 0.185 | 0.874 | 0.961 | 0.983 |
| 8 | 0.116 | 0.934 | 4.857 | 0.191 | 0.874 | 0.959 | 0.981 |
| 9 | 0.110 | 0.820 | 4.667 | 0.187 | 0.879 | 0.961 | 0.982 |
| 10 | 0.113 | 0.896 | 4.757 | 0.188 | 0.878 | 0.960 | 0.982 |
| 11 | 0.112 | 0.883 | 4.717 | 0.188 | 0.881 | 0.961 | 0.982 |
| 12 | 0.111 | 0.853 | 4.751 | 0.187 | 0.878 | 0.961 | 0.982 |
| 13 | 0.112 | 0.842 | 4.640 | 0.186 | 0.882 | 0.962 | 0.982 |
| 14 | 0.110 | 0.822 | 4.642 | 0.184 | 0.882 | 0.962 | 0.983 |
| 15 | 0.109 | 0.805 | 4.612 | 0.184 | 0.882 | 0.962 | 0.983 |
| 16 | 0.108 | 0.822 | 4.595 | 0.183 | 0.885 | 0.962 | 0.983 |
| 17 | 0.109 | 0.833 | 4.633 | 0.185 | 0.883 | 0.962 | 0.982 |
| 18 | 0.109 | 0.845 | 4.661 | 0.187 | 0.884 | 0.961 | 0.982 |
| 19 | 0.107 | 0.795 | 4.580 | 0.184 | 0.886 | 0.962 | 0.983 |
| 20 | 0.108 | 0.825 | 4.631 | 0.184 | 0.885 | 0.962 | 0.983 |
| 21 | 0.110 | 0.867 | 4.653 | 0.186 | 0.884 | 0.961 | 0.982 |
| 22 | 0.108 | 0.844 | 4.635 | 0.185 | 0.886 | 0.962 | 0.982 |
| 23 | 0.107 | 0.815 | 4.629 | 0.185 | 0.887 | 0.962 | 0.982 |
| 24 | 0.108 | 0.836 | 4.626 | 0.184 | 0.886 | 0.962 | 0.982 |
| 25 | 0.107 | 0.813 | 4.620 | 0.184 | 0.887 | 0.962 | 0.982 |
| 26 | 0.108 | 0.827 | 4.624 | 0.184 | 0.886 | 0.962 | 0.982 |
| 27 | 0.108 | 0.820 | 4.608 | 0.184 | 0.886 | 0.962 | 0.982 |
| 28 | 0.108 | 0.832 | 4.622 | 0.184 | 0.887 | 0.962 | 0.982 |
| 29 | 0.107 | 0.821 | 4.608 | 0.184 | 0.887 | 0.962 | 0.982 |

## Interpretation

- What improved: AbsRel·RMSE log·δ₁~δ₃는 논문과 ±0.001 안에 들어온다. δ₁ 0.887은 논문(0.886)보다 약간 높다.
- What became worse: SqRel 0.821 (논문 0.765, +7%), RMSE 4.608 (논문 4.561, +1%). 큰 오차 쪽이 약간 크다.
- 안정성: epoch 19 이후 AbsRel 0.107~0.110 범위에서 수렴. SqRel은 epoch마다 0.79~0.87로 흔들려 run 간 편차도 이 정도는 있을 것으로 보인다. seed 없는 run 하나라 차이가 의미 있는지는 판단 불가.
- Failure cases (2026-11-23, `analysis/baseline_failure_analysis.py`, 상세: `outputs/.../analysis/weights_29/summary.md`):
  - 평균을 재계산하면 evaluate_depth.py와 같다 (0.107 / 0.821 / 4.608 …), 프로토콜 일치 확인.
  - 이미지별 AbsRel: median 0.098, p90 0.162, max 0.291. 최악 10%(70장)가 AbsRel 합의 17.9%. 소수 이미지가 평균을 끌어올리는 구조는 아니다.
  - **깊이 불연속 근처(GT 9×9 안에서 깊이 1.25배 이상 차이)**: 점의 11.5%인데 AbsRel 합의 28.1%. AbsRel 0.260 vs 나머지 0.086 (3배). 이미지별 AbsRel과의 Spearman +0.54로 조사한 요인 중 가장 강하다. LiDAR·카메라 시차 artifact도 섞여 있어 정성 확인이 필요하다.
  - **거리**: 0–20m 0.088 / 20–40m 0.155 / 40–80m 0.201. 40m 이상은 평균적으로 너무 가깝게 예측(mean log(pred/gt) −0.147). 다만 점의 5.7%, AbsRel 합의 10.8%.
  - 정성 유형: (a) 물체·나무 기둥·차량 윤곽의 경계 번짐, (b) 트인 장면에서 지평선 근처 수풀·가로수 띠를 너무 멀게 예측(idx 152, 164), (c) 큰 건물 벽면을 너무 가깝게 예측(idx 549, ≥40m에서 log −0.94), (d) 철망·덤불처럼 뒤가 비치는 구조물(drive 0086, idx 395/394/374)에서 LiDAR는 뒤를, 모델은 앞면을 봐서 오차가 섞임(GT 모호성 포함).
  - 최상 5장은 모두 drive 0002(도로·근거리 위주).
  - 경계 양쪽 분리 (2026-11-23): 배경 쪽 AbsRel 0.293, log(pred/gt) −0.348 (29% 가깝게), δ₁ 0.396 / 앞 물체 쪽 0.282, +0.162 (18% 멀게). 깊이가 경계를 넘어 섞이는 번짐(edge fattening).
  - 상위 30% 209장 검사 (`analysis/inspect_worst_fraction.py`, `.../weights_29/worst30/`): 상위 30%는 나머지보다 경계(0.329 vs 0.221)뿐 아니라 평탄 영역(0.129 vs 0.070)·0–20m(0.139 vs 0.067)에서도 오차가 크다. 오차 중 경계 비중은 비슷하다(29.9% vs 26.8%). 즉 경계 번짐은 모든 이미지의 공통 약점이고, 최악 이미지는 장면 전체 구조 오류가 겹친 경우다.
  - 209장을 직접 본 경향 (Claude가 확인, 이미지별 라벨은 아직 없음, `labels_template.csv`):
    - 주차·정차 차량 윤곽, 특히 윗선을 따라 배경을 가깝게 예측하는 띠: 차량이 있는 이미지 대부분
    - 대형 트럭·정체 차량 장면: drive 0052가 25장 중 22장
    - 보행자·자전거가 많은 장면: 0071, 09_28_0002
    - 고속도로·트인 장면에서 먼 수풀 띠를 너무 멀게 예측: 0029, 10_03_0047, 0101
    - 철망·덤불처럼 뒤가 비치는 구조물: 0086
    - 앞 차량 일부를 너무 멀게 예측: 같은 방향으로 움직이는 물체 문제로 추정
  - 투과 점 분리 (2026-11-27): 경계 배경 쪽 점 중 같은 줄의 좌우 20px 안에 더 가까운 점이 모두 있는 점(유리창·철망 너머, GT 모호)은 오차의 1.2%뿐이다. 윤곽 바깥 배경 쪽이 12.3% (AbsRel 0.288, log −0.338). 다만 차 유리창은 20px보다 넓어서 이 판정을 일부 빠져나간다.
  - 픽셀 단위 경계 위치 (`analysis/edge_offset.py`, `.../weights_29/edges/summary.md`): 양쪽이 일관된 면인 LiDAR 깊이 경계 3,263개(645장)를 대상으로 했다.
    - 정확(±4 GT px 이내) 25.4%
    - 배경 쪽으로 >4px 밀림(번짐) 27.7%
    - 앞 물체 쪽으로 >4px 밀림 7.1%
    - 앞 물체를 놓침 22.9%
    - 배경에 끝까지 닿지 않음 16.9% (주로 유리·철망 너머로 추정)
    - 위치를 측정한 경계의 중앙값 +3.5px. 배경 쪽 대 앞 물체 쪽 비대칭이 약 4:1이라 앞 물체가 바깥으로 부푼다.
  - 보고서용 그림 (`analysis/boundary_zoom_figures.py`, `.../weights_29/boundary_zoom/edge/`): 밀린 거리 8–24px, 깊이 비 ≥2, drive당 1개 규칙으로 idx 213, 364, 68, 480, 529를 골랐다. `abs_rel/`은 AbsRel 최악 5장(395, 152, 394, 549, 164)이다.
- Deviations from the planned protocol: PNG 입력(논문 JPG), seed 없음, num_workers 6, 평가 코드의 `--png` 무시 버그 수정 (CHANGES.md 참고). 공개 checkpoint는 JPG로 학습됐지만 PNG 입력으로 평가해도 논문 수치가 그대로 나온다.
- Next action: 이 run을 로컬 baseline으로 사용. SqRel 차이가 PNG/JPG 때문인지 run 편차인지 보려면 JPG 학습 또는 seed 고정 반복 실행이 필요하다.
