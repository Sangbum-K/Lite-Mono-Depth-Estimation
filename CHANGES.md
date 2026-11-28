# 논문 재현과 다른 점 (CHANGES)

Lite-Mono 논문(CVPR 2023)과 공식 코드 설정에서 **이 프로젝트가 달라진 모든 것**을 기록한다.
구현 환경, 데이터, 학습 설정, 코드 수정, 실행 기록을 포함한다. 무엇이든 바꾸면 이 파일을 함께 고친다.

- 기준(=논문 방식): 논문 본문 + 공식 저장소 github.com/noahzn/Lite-Mono commit `4874b35`의 README 학습 명령 + monodepth2 데이터 준비 방식
- 실행 환경 두 가지: **로컬** (이 PC, `main`/`Local` 브랜치) / **Colab** (`origin/Colab` 브랜치)
- 표기: ✅ 논문과 같음 · ⚠️ 다름(결과에 영향 가능) · ➖ 다르지만 결과에 영향 없음

---

## 1. 구현 환경

| 항목 | 논문/공식 | 로컬 | Colab |
|---|---|---|---|
| GPU | 학습: NVIDIA TITAN Xp 1장, batch 12 (논문 4.1절 원문 확인 11-28). 속도 측정: TITAN Xp와 Jetson Xavier (논문 4.5절) | RTX 5070 12GB | Colab 할당 GPU (세션마다 다름, `gpu_log.csv`에 기록) |
| Driver / CUDA | 기재 없음 | 580.178.04 / CUDA 12.8 | 세션마다 다름 |
| OS | 기재 없음 | Ubuntu 24.04.4 LTS | Colab 기본 이미지 |
| CPU / RAM | 기재 없음 | 16 threads / 15GB | Colab 할당 |
| Python | 기재 없음 (monodepth2 계열은 3.6~3.8대) | 3.11.15 (`.venv`) | Colab 기본 |
| PyTorch / torchvision | 구버전 (정확한 버전 기재 없음) | 2.11.0+cu128 / 0.26.0+cu128 | Colab 기본 (pip freeze로 기록) |
| NumPy / Pillow | 구버전 | 1.26.4 / 12.2.0 | Colab 기본 |
| 기타 패키지 | README의 pip 목록 | (분석 전용 추가: matplotlib 3.11.2, 2026-11-23) timm 1.0.31, tensorboardX 2.6.5, thop 0.1.1, scikit-image 0.26.0, opencv-python-headless 4.11.0.86 | `environment/requirements-colab.txt` (버전 미고정) |
| LR scheduler 패키지 | README: git 저장소 최신 | commit `73bf958` 고정 | commit `73bf958` 고정 |

➖ 라이브러리가 새 버전이라 일부 코드 수정이 필요했다(4장). GPU·라이브러리 차이로 수치가 완전히 같게 나오지는 않을 수 있다.

## 2. 데이터

| 항목 | 논문/공식 | 로컬 | Colab |
|---|---|---|---|
| 데이터셋 | KITTI raw, Eigen split | ✅ | ✅ |
| 학습/검증 목록 | eigen_zhou 39,810 / 4,424 (논문 본문은 39,180으로 기재, 오타 추정) | ✅ monodepth2 목록 | ✅ monodepth2 목록 |
| 테스트 목록 | eigen 697장 | ✅ | ✅ |
| **이미지 포맷** | **JPG** (monodepth2 방식: PNG → `convert -quality 92 -sampling-factor 2x2,1x1,1x1`) | ⚠️ **원본 PNG** (`--png`) | ✅ JPG, 같은 설정 (`scripts/prepare_kitti.py`) |
| 데이터 위치 | — | `data/raw/kitti/` | Drive → `/content` 복원 |

- ⚠️ 로컬의 PNG 사용: 변환 단계 없이 받은 원본을 바로 쓰려고 정한 것으로 보인다(기록 없음). 입력 화질이 논문과 달라서 결과에 영향을 줄 수 있다.
- val 4,424장 중 3,950장은 train에 있는 같은 순간을 반대쪽 카메라로 찍은 영상이다. 그래서 val 점수는 낙관적일 수 있다(원본 split 자체의 특성).
- ➕ 논문에 없는 평가 전용 데이터: KITTI raw tracklet(`<drive>_tracklets.zip`)을 Eigen test 주행 중 공개된 17개(모두 2011_09_26)만 받았다. 저장 위치는 `data/raw/kitti_archives/`(zip)와 `data/raw/kitti/<date>/<drive>_sync/tracklet_labels.xml`이다. 0093 zip은 폴더 없이 파일만 들어 있어 위치를 옮겼다. 학습에는 쓰지 않는다. 11-28에 이 데이터로 만든 커버리지 결과는 임의 스크립트라 리포트에서 빼고 스크립트·출력을 제거했으며, 원본 데이터만 남겨 두었다.

## 3. 학습 설정

| 항목 | 논문/공식 | 로컬 (`scripts/train_baseline.sh`) | Colab (노트북 G 셀) |
|---|---|---|---|
| 모델 | lite-mono (3.1M) | ✅ | ✅ |
| 해상도 | 640×192 | ✅ | ✅ |
| batch / epochs | 12 / 30 (`options.py` 기본값 16 / 50과 다름) | ✅ 명시 | ✅ 명시 |
| lr | `0.0001 5e-6 31 0.0001 1e-5 31` | ✅ | ✅ |
| encoder 초기화 | ImageNet pretrain (`lite-mono-pretrain.pth`) | ✅ | ✅ |
| seed | 없음 | ✅ 없음 | ➖ `--seed 42` (추가 기능, 비교 실험용) |
| num_workers | README 명령에 없음 → 기본값 12 | ➖ 6 (RAM 15GB에서 12는 OOM) | ➖ 8 |
| 이어서 학습 | 없음 | 없음 | ➖ 지원 (LR 스케줄까지 복원) |

- seed와 num_workers는 같은 seed 안에서 증강 난수 순서를 바꾼다. 로컬과 Colab 결과를 1:1로 비교하려면 둘을 맞춰야 한다.
- seed가 없는 run은 같은 명령으로 다시 돌려도 결과가 달라진다.

## 4. 공식 코드 수정 (`vendor/Lite-Mono/`)

### 4.1 호환 수정 — 결과에 영향 없음 ➖

| 파일 | 문제 | 로컬 | Colab |
|---|---|---|---|
| `datasets/mono_dataset.py` | Pillow 10에서 `Image.ANTIALIAS` 삭제 | `InterpolationMode.LANCZOS` | ⚠️ **파일 자체가 브랜치에 없음** (6장) |
| `networks/resnet_encoder.py` | torchvision에서 `model_urls` 삭제 | `ResNet18_Weights.IMAGENET1K_V1`, `weights=` 인자 | `get_model_weights(...)` (같은 파일) |
| `trainer.py` (pretrain 로드) | torch 2.6부터 `weights_only=True` 기본값 | `safe_globals`로 필요한 타입만 허용 | `weights_only=False` (체크섬 확인한 파일만) |
| `trainer.py` (val) | `iterator.next()` 삭제 | `next(iterator)` | 같음 |
| `kitti_utils.py` | NumPy 1.24에서 `np.int` 삭제 | `int` | 같음 |
| `export_gt_depth.py` | 최신 NumPy가 크기 다른 배열 묶음 거부 | Colab 브랜치 파일을 그대로 가져옴 (2026-11-09) | 수정해서 추가 |

로컬 수정 내역 전체: `vendor/changes_from_upstream.patch` (upstream `4874b35` 대비 수정 5개 파일 + 추가 `export_gt_depth.py`). 중첩 `.git`은 저장소 밖 `/home/user/sangbum/Lite-Mono-upstream.git`으로 옮겼다.

### 4.1b 버그 수정 — 평가 입력에만 영향 ⚠️

| 파일 | 문제 | 로컬 | Colab |
|---|---|---|---|
| `evaluate_depth.py` | `--png`를 줘도 평가 데이터셋에 `img_ext`를 넘기지 않아 항상 `.jpg`를 읽음 (monodepth2부터 있던 upstream 버그) | `img_ext='.png' if opt.png else '.jpg'` 전달 (2026-11-09) | 수정 없음 (JPG 사용이라 문제 없음) |

- 로컬은 PNG만 있어서 이 수정 없이는 평가가 FileNotFoundError로 실패한다. 계산 방식(median scaling, 80m, Garg crop)은 그대로다.

### 4.2 기능 추가

| 기능 | 로컬 | Colab | 결과 영향 |
|---|---|---|---|
| `--seed` | 없음 | 있음 | ➖ 안 쓰면 원본과 같음 |
| 이어서 학습 (`train_state.json`) | 없음 | 있음 | ➖ 계산은 같음. 이어서 한 run은 데이터 순서가 끊김 없는 run과 달라짐 |
| checkpoint 정리 | 없음 (모든 epoch 전부 저장) | pose/optimizer 파일은 최신 것만 저장 | ➖ |
| 진행률·남은 시간 출력 | 없음 | 있음 | ➖ |
| pretrain 로드 메시지 | 불러온 텐서 수 출력 (`233/263`) | 원본 그대로 | ➖ |
| `--hires_decoder` (개선안 A, 2026-11-27) | 있음: encoder가 1/2 해상도 stem 특징도 반환(`return_stem`), `HiResDepthDecoder`가 마지막 단계에 그 skip을 붙이고 scale-0 disparity를 학습형 2× transposed conv로 만듦(+11,848 파라미터, +0.39%) | 없음 | ⚠️ 켜면 모델 구조가 바뀜(실험용). 끄면 원본과 같음: 공개 checkpoint 출력이 완전히 같은 것 확인 |
| `--hires_stem_ch N`, `--hires_upsample {deconv,subpixel}` (개선안 A-lite, 2026-11-27) | 있음: `--hires_decoder`와 함께 사용. stem skip(48채널)을 1×1 conv+ELU로 N채널로 압축, scale-0 head를 `Conv3x3(c0,4)`+`PixelShuffle(2)`(SuperDepth의 sub-pixel conv)로 교체. pilot 설정 N=8+subpixel: 5.117 GMACs(+1.7%), 3.0713M(+0.09%) (thop, 640×192). 참고 A: 5.550G(+10.3%) | 없음 | ⚠️ 켜면 구조가 바뀜. 기본값(0, deconv)이면 기존 A와 같음(state_dict 키 동일) |
| `--smooth_edge_gain g` (개선안 C, 2026-11-27) | 있음: edge-aware smoothness 가중치를 `exp(-|∇I|)` → `exp(-g·|∇I|)`. 학습 영상 150장 측정: 상위 5% 밝기 경계(|∇I|=0.13)에서도 원래 가중치가 0.88이라 경계에서 거의 안 풀림. pilot은 g=10(같은 경계 0.27, 중앙값 영역 0.88). 값은 이 분포만 보고 정했고 튜닝하지 않음. 처음 측정은 스크립트 미보관이라 리포트에서는 수치를 빼고 식으로만 설명(11-28) | 없음 | ⚠️ 학습 손실만 바뀜. 기본값 1.0이면 원본과 비트 단위로 같은 값 확인 |
| `--dpt_decoder`, `--dpt_dim N`, `--dpt_head_ch M` (개선안 E, 2026-11-28) | 있음: `DPTDepthDecoder`. encoder는 그대로 두고 decoder 전체를 Depth Pro 2의 DPT decoder 구조로 바꾼다(단계마다 residual 정제 + 학습형 2× transposed conv, head도 학습형 확대). 출력은 Lite-Mono와 같은 sigmoid disparity, scale 0–2. 채널 24·head 16에서 5.133G(+2.0%), 2.970M(−3.42%) (`analysis/model_cost.py`, thop, 640×192). Depth Pro 2-S 채널(64)은 +81.9% | 없음 | ⚠️ 켜면 decoder가 바뀜(실험용). 끄면 원본과 같음(기존 경로는 수정하지 않음). `--hires_decoder`와 함께 쓸 수 없음 |
| `--val_num_workers N` (2026-11-27) | 있음: 검증 loader의 worker 수만 따로 지정(-1 = `--num_workers`와 같음). pilot은 1 사용 | 없음 | ➖ 표본 순서는 메인 프로세스 sampler가 정하므로 학습 계산은 같음. RAM만 줄어듦 |

### 4.3 upstream에 없어서 추가한 파일 (monodepth2 commit `19e16e1`)
- `splits/eigen_zhou/{train,val}_files.txt`: 로컬 ✅ / Colab ✅
- `export_gt_depth.py`, `LICENSE_monodepth2`: 로컬 ✅ (2026-11-09, Colab 브랜치에서 가져옴) / Colab ✅
- `splits/eigen/gt_depths.npz`: 로컬에서 PNG와 같은 velodyne 원본으로 생성 (`*.npz` gitignore라 커밋 안 됨, `export_gt_depth.py --split eigen`으로 재생성)

## 5. 실행 기록

| run | 환경 | 날짜 | 조건 | 논문과 다른 점 | 상태 |
|---|---|---|---|---|---|
| `20261109_baseline_unseeded` | 로컬 | 2026-11-09 시작 | 공식 하이퍼파라미터, ImageNet 초기화 | ⚠️ PNG 입력, seed 없음, num_workers 6 | 완료 (2026-11-09 22:09, 약 9시간). Eigen test `weights_29`: AbsRel 0.107 / SqRel 0.821 / RMSE 4.608 / RMSElog 0.184 / δ₁ 0.887 / δ₂ 0.962 / δ₃ 0.982. 상세: `experiments/20261109_baseline_unseeded.md` |
| `20261127_smooth1e-4_e5` (B) | 로컬 | 2026-11-27 18:28–19:36 | 5 epoch, `--disparity_smoothness 1e-4` (원래 1e-3) | ⚠️ 가설 2 pilot, num_workers 6, val workers 1, OMP 1 | 완료. AbsRel 0.114, 번짐 31.3% (`outputs/pilots_summary.md`) |
| `20261127_hires_lite_e5` (A-lite) | 로컬 | 2026-11-27 19:36–20:44 | 5 epoch, `--hires_decoder --hires_stem_ch 8 --hires_upsample subpixel` | ⚠️ 가설 1 pilot, 같은 worker 설정 | 완료. AbsRel 0.116, 번짐 32.9% |
| `20261127_smoothgain10_e5` (C) | 로컬 | 2026-11-27 20:44–21:53 | 5 epoch, `--smooth_edge_gain 10` | ⚠️ 가설 2 pilot, 같은 worker 설정 | 완료. AbsRel 0.127, 번짐 26.7% |
| `20261127_hires_decoder_e5` (A) | 로컬 | 2026-11-27 21:53–23:01 | 5 epoch, `--hires_decoder` | ⚠️ 가설 1 pilot, 같은 worker 설정 (이전 시도는 OOM으로 중단 후 폴더 삭제) | 완료. AbsRel 0.116, 번짐 27.8% |
| `20261128_baseline_rerun_e5` (T1) | 로컬 | 2026-11-28 00:25–01:35 | 5 epoch, baseline과 같은 설정 | ⚠️ 학습 간 노이즈 측정용, 같은 worker 설정 | 완료. ep5 AbsRel 0.117 / SqRel 0.815 / RMSE 4.674, 번짐 25.1%(원래 baseline ep5 28.7%) |
| `20261128_dpt24_e5` (E) | 로컬 | 2026-11-28 15:12–16:22 | 5 epoch, `--dpt_decoder --dpt_dim 24 --dpt_head_ch 16` | ⚠️ 가설 1 추가 pilot(Depth Pro 2 decoder 구조), 같은 worker 설정 | 완료. ep5 AbsRel 0.114 / SqRel 0.771 / RMSE 4.796, 번짐 29.5%(ep4 30.3%), 물체 놓침 23.7%(ep4 25.5%) |
| 실패 분석 `weights_29` | 로컬 | 2026-11-23 | `analysis/baseline_failure_analysis.py`, 이미지별·거리별·영역별 오차와 고정 규칙 샘플 그림 | ➖ 평가 프로토콜 그대로 (평균 재계산 일치) | 완료. 결과: `outputs/20261109_baseline_unseeded/analysis/weights_29/summary.md`, 요약은 실험 기록 |
| 공개 checkpoint 평가 | 로컬 | 2026-11-09 | `pretrained/kitti_640x192`, 학습 없음 | ⚠️ PNG 입력으로 평가 (checkpoint는 JPG로 학습됨) | 완료. AbsRel 0.107 / SqRel 0.766 / RMSE 4.560 / RMSElog 0.183 / δ₁ 0.886 / δ₂ 0.963 / δ₃ 0.983 (논문과 같음). 로그: `outputs/public_kitti_640x192_eval/eval_png.log` |

- 이 run은 PNG 입력이라 논문과 조건이 완전히 같지는 않다. 그래도 AbsRel·RMSE log·δ는 논문과 ±0.001 안이고, SqRel(+0.056)과 RMSE(+0.047)만 약간 높다. 로컬 개선 실험의 baseline으로 쓴다.
- SqRel 차이가 PNG 입력 때문인지 run 편차(seed 없음) 때문인지는 아직 모른다. epoch 19~29 사이에서도 SqRel이 0.795~0.867로 흔들렸다.

## 6. 알려진 문제

- ⚠️ **Colab 브랜치에 `vendor/Lite-Mono/datasets/` 폴더가 빠져 있다.** 루트 `.gitignore`의 `datasets/` 규칙(KITTI 데이터 제외용)이 코드 폴더까지 막았다. 이대로면 Colab에서 clone 후 `train.py`가 `import datasets`에서 실패한다. 또 CHANGES에 적힌 `mono_dataset.py`의 ANTIALIAS 수정도 실제로는 커밋되지 않았다. → `.gitignore`를 `/datasets/`처럼 루트 기준으로 좁히거나 `!vendor/Lite-Mono/datasets/`를 추가한 뒤 커밋해야 한다.
- `Local` 브랜치에서는 `.gitignore`를 `/datasets/`로 고쳐 해결했다. Colab 브랜치에는 아직 반영되지 않았다.

## 7. 아직 확인 못 한 것

- 공개 checkpoint 재현: 로컬 PNG 입력으로는 확인됨(5장). Colab JPG 입력은 미확인.
- 논문의 학습 GPU·소프트웨어 버전 원문 표기.
- PNG와 JPG 학습의 실제 성능 차이 (특히 SqRel 0.821 vs 0.765).

---

## 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-11-08 ~ 09 | (Colab 브랜치) 공식 코드 반입, 호환 수정, Colab 이어서 학습·seed 기능, JPG 준비 스크립트 |
| 2026-11-09 | (로컬) 공식 코드 반입·호환 수정, PNG·unseeded baseline 학습 시작 |
| 2026-11-09 | 로컬·Colab을 함께 다루는 CHANGES.md 작성. Colab 브랜치 `datasets/` 누락 발견 |
| 2026-11-09 | 로컬 코드를 `Local` 브랜치에 커밋. `.gitignore` `datasets/` → `/datasets/`, 로컬 수정 patch 저장 |
| 2026-11-09 | (로컬) baseline 학습 완료·Eigen test 평가 (AbsRel 0.107). `evaluate_depth.py` `--png` 버그 수정, `export_gt_depth.py` 반입, `scripts/eval_eigen.sh` 추가, 공개 checkpoint 평가 |
| 2026-11-23 | (로컬) baseline 실패 분석 스크립트·결과 추가, 분석용 matplotlib 설치 |
| 2026-11-23 | (로컬) 상위 30% 검사 스크립트, 경계 양쪽 분리 분석 추가. Depth Pro 2 저장소를 `third_party/`에 받음(gitignore, 학습·평가 코드에는 사용 안 함), 분석 노트 `research/depth_pro_2_notes.md` |
| 2026-11-27 | (로컬) 경계 픽셀 단위 위치 분석(`analysis/edge_offset.py`), 보고서용 경계 확대 그림(`analysis/boundary_zoom_figures.py`), 투과 점 분리를 분석 요약에 추가 |
| 2026-11-27 | (로컬) 보고서용 A4 경계 그림 `boundary_zoom/edge/summary_a4.{png,pdf}` 생성 (`analysis/boundary_zoom_figures.py --select edge`) |
| 2026-11-27 | (로컬) 보고서용 경계 비교 그림을 사진 위 경계 표시 방식으로 교체: `analysis/boundary_overlay_figure.py` → `boundary_overlay/overlay_a4.{png,pdf}` |
| 2026-11-27 | (로컬) idx 364 단일 경계 그림(LiDAR-예측 겹침 패널 포함): `boundary_overlay_figure.py --single 364` → `boundary_overlay/single_idx364.{png,pdf}` |
| 2026-11-27 | (로컬) 경계 그림에서 LiDAR 기반 "true edge" 초록 점 제거 (유리 투과·노이즈 때문에 경계 정답으로 보기 어려움) |
| 2026-11-27 | (로컬) `boundary_zoom/edge/idx*_zoom.png`의 (h) 패널을 경계 종류 지도에서 "예측 위 LiDAR 점" 겹침으로 교체 (`--select edge --crop_h 72`) |
| 2026-11-27 | (로컬) `idx*_zoom.png`에서 (i) 단면 그래프와 관련 표시(사진 위 점선, 규칙 기반 경계 수치) 삭제, `idx*_profile.png` 생성 중단 |
| 2026-11-27 | (로컬) Depth Pro 2 기준선 비교: 분리 환경 `third_party/.venv-dp2`(메인 `.venv`에 av·pillow-heif 추가 안 함), `analysis/run_depth_pro_2.py`, `compare_models.py`, `compare_sheet.py`. `baseline_failure_analysis.disp_at` 추가(원본 해상도 예측은 보간 없이 crop, Lite-Mono 결과 불변 확인) |
| 2026-11-27 | (로컬) Depth Pro 2-H+ 추론·비교 추가 (`outputs/depth_pro_2/h`, `compare/`) |
| 2026-11-27 | (로컬) 개선안 A `--hires_decoder` 구현(`networks/depth_encoder.py`, `depth_decoder.py`, `options.py`, `trainer.py`, `evaluate_depth.py`), patch 갱신. 5 epoch pilot `20261127_hires_decoder_e5` 시작 (첫 시도는 분석 작업과 동시 실행해 RAM OOM으로 중단, 단독으로 다시 실행) |
| 2026-11-27 | (로컬) A안 학습이 단독 실행에서도 RAM OOM(16:33)으로 중단. 사용자가 터미널에서 `--num_workers 4`로 직접 실행하기로 함(데이터 로딩 worker 수만 다름, 학습 계산은 같음). `baseline_failure_analysis.py`에 `--hires_decoder` 옵션과 decoder 가중치 일치 검사 추가 |
| 2026-11-27 | (로컬) 중간 보고서 초안 `reports/HW2_Intermediate_Report_draft.docx` 작성 (proposal docx 서식 재사용). 그림은 기존 분석 그림을 잘라 붙임: `analysis/intermediate_report_figures.py` → `reports/figures/intermediate/`. 원고 `reports/intermediate_draft.md`(직접 작성), docx 서식 변환 `reports/intermediate_src/build_intermediate.py`. A안 pilot 결과 칸은 비워 둠(TBD) |
| 2026-11-27 | `.gitignore`: `outputs/` 안의 그림(png/pdf)만 git에 포함하도록 변경(93개, 약 100MB). checkpoint·로그·csv는 계속 제외 |
| 2026-11-27 | (로컬) A-lite(`--hires_stem_ch`, `--hires_upsample subpixel`)·C(`--smooth_edge_gain`) 구현, `evaluate_depth.py`·`analysis/baseline_failure_analysis.py`에 같은 옵션 연결, patch 갱신(fresh upstream `4874b35`에 적용 후 vendor와 파일 동일 확인). 4개 설정 모두 Trainer로 2 batch 학습 smoke test, A-lite 더미 checkpoint로 평가·분석·edge_offset 경로 확인 |
| 2026-11-27 | (로컬) pilot 일괄 실행 `scripts/run_pilots.sh`(B → A-lite → C → A, 한 번에 하나, 끝난 것은 건너뜀), 결과 표 `analysis/pilot_summary.py` → `outputs/pilots_summary.md` |
| 2026-11-27 | (로컬) 중간 보고서 원고 `reports/intermediate_draft.md` 6장을 가설 2개(decoder 구조 / 자기지도 손실) + 후보 A·A-lite·B·C·D + 선행 연구 점검(HR-Depth, SuperDepth, Chen 등 WACV 2023, BoRe-Depth)으로 다시 씀. 5.4절의 "구조가 중요" 서술을 DP2가 지도학습이라 원인을 나눌 수 없다는 내용으로 수정 |
| 2026-11-27 | (로컬) pilot 첫 실행(18:13): `run_pilots.sh`가 거의 동시에 두 번 실행된 것으로 보이며, 학습 두 개가 겹쳐 RAM OOM → B·A-lite·C가 시작 30초 안에 종료됨. A만 단독으로 계속 학습. `run_pilots.sh` 개선: 중복 실행 잠금(flock)과 다른 train.py 실행 중이면 시작 거부, OOM 시 worker 2개로 한 번 재시도 후 실패하면 전체 중단(다음 pilot로 넘어가지 않음), 단계별 로그. 진행 확인 `scripts/pilot_status.py [--watch]` 추가. 실행 중인 스크립트를 제자리 수정한 실수가 있어, 실행 중 bash가 읽는 원래 내용을 복원하고 새 버전은 새 파일로 교체(mv) |
| 2026-11-27 | (로컬) RAM 분석: 학습 하나가 약 7GB 사용(메인 1.6GB + 미리 읽어 둔 batch 공유 메모리 4.2GB + worker 8개). 검증 worker 4개는 250 batch마다 한 번만 쓰이는데 batch를 계속 미리 읽어 둠 → `--val_num_workers` 추가, pilot은 1로 실행. 사용자 요청으로 실행 중이던 pilot(A, 이어서 B)을 모두 종료하고 4개 pilot 출력 폴더를 지운 뒤 18:2x에 처음부터 다시 시작 |
| 2026-11-27 | (로컬) GPU가 데이터 대기로 절반 쉼(0.5초 간격 75%↔4%, GPU 계산 약 58장/s vs 실제 약 26장/s). 원인: CPU 데이터 준비(PNG 디코딩 + 4단계 Lanczos resize, worker당 약 150ms/샘플), worker당 스레드 33개로 load 30/16. `run_pilots.sh` 기본 worker 4→6(검증 1 유지), `OMP/OPENBLAS/MKL_NUM_THREADS=1`. 학습 계산은 같고 속도만 바뀜. 사용자 승인 후 pilot 다시 처음부터 시작 |
| 2026-11-28 | (로컬) pilot 4개 완료(18:28–23:01, 실험당 약 68분, OOM 없음) → `outputs/pilots_summary.md`. 자율주행 관점 지표 `analysis/ad_metrics.py` 추가 → `outputs/pilots_ad.md`: 저장된 예측(disps_eigen.npy)·edges.csv만 사용, LiDAR 20m 이내를 0–10m/10–20m로 나누어 AbsRel, 실제보다 25% 이상 멀게/가깝게 예측한 비율, 물체 쪽 경계가 멀게 예측된 비율, 물체 옆 빈 공간이 막힌 비율, 경계 결과를 계산. baseline ep5와의 차이에 테스트 영상 단위 paired bootstrap 95% 구간(학습 무작위성은 포함하지 않음) |
| 2026-11-28 | (로컬) 노이즈 확인용 추가 분석(학습 없음): baseline weights_5·6·9·14·19·24, 그리고 baseline과 pilot 4개의 weights_3(ep4)에 `baseline_failure_analysis.py --no_figs` + `edge_offset.py` 실행 → `outputs/<run>/analysis/weights_N/`. 결과: 인접 epoch 사이 변화(예: baseline ep5→ep6 0–10m AbsRel 0.097→0.088)가 후보 간 차이보다 큼, 배경 쪽 번짐은 ep5–ep30 내내 26–30%로 학습이 진행돼도 줄지 않음, 경계 4px 이내 비율은 ep4–5에서 모든 실험이 18–21%로 같음 |
| 2026-11-28 | (로컬) T1: baseline 5 epoch 재학습(같은 설정, 학습 노이즈 측정) `run_pilots.sh Base` → `outputs/20261128_baseline_rerun_e5` (00:25 시작). `run_pilots.sh`·`pilot_status.py`·`pilot_summary.py`에 Base 항목 추가(이름을 지정할 때만 실행) |
| 2026-11-28 | (로컬) V2 손실 진단 `analysis/loss_preference.py` 추가(학습 없음): baseline ep30의 pose net과 학습과 같은 photometric loss(640×192, SSIM 3×3 + L1, 원본 프레임 최솟값, identity loss)로, LiDAR 경계 주변 픽셀에 물체 깊이와 배경 깊이를 줬을 때 손실을 비교 → `outputs/20261109_baseline_unseeded/analysis/weights_29/loss_preference.md`. 결과: 경계 옆 배경 1–2 px에서 손실이 물체 깊이를 고르는 비율 50%, 3–4 px 35%, 9–16 px 26%(기저). ±2 프레임 추가해도 거의 같음. 손실이 물체 깊이를 고르는 경계에서 모델 번짐 35% vs 24%, 밝기 대비가 큰 경계에서 40% vs 25% |
| 2026-11-28 | (로컬) 사용자 결정으로 추가 학습(DP2 pseudo-label 등)은 하지 않고 리포트 정리. `reports/intermediate_draft.md`: 초록에 예비 실험·손실 진단 결과 추가, 6.3절 본인 설계에 손실 진단 추가, 6.5절 예비 실험 표(5번째·4번째 epoch, baseline 재학습 열)와 해석 작성, 6.6절 손실 함수 진단 신설, 6.7절 향후 계획(후보 D 우선, 경계 주변 추가 신호, 다중 checkpoint·다중 학습 비교) 재작성. docx 재빌드 |
| 2026-11-28 | 전체 실험 기록 `experiments/EXPERIMENT_LOG.md` 작성(E0–E12: 왜 했는지, 사전 가설, 방법, 결과, 확인한 것, 한계, 해석 정정 기록, 가설 상태 요약). `experiments/README.md`에 안내 추가. CHANGES 5장 실행 기록 표의 pilot 상태를 실제 실행(시각, worker 설정, 결과)으로 갱신 |
| 2026-11-28 | (로컬) T1 완료 후 재학습 baseline의 weights_3도 분석. 리포트 6.5절 표의 재학습 열과 해석(두 baseline 학습 범위를 두 checkpoint 모두에서 벗어난 것은 A-lite의 번짐·물체 놓침뿐) 채움, docx 재빌드. `EXPERIMENT_LOG.md` E11 결과 기록 |
| 2026-11-28 | (로컬) 리포트에 6.7절 결과 종합 신설(가설별 확인 방법·결과·판정 표, 종합 결론, 해석을 고친 경우). 기존 향후 계획은 6.8절로 이동. 서론 마지막 문장에 예비 실험 결과 추가. docx 재빌드(XML 정상). 6.4절 tracklet 커버리지 1단계 확인: Eigen test 28개 주행 중 tracklet 공개 17개(2011_09_26, 422장), 물체 수 집계는 미완 |
| 2026-11-28 | (로컬) 6.4절 tracklet 커버리지 2단계 완료: `analysis/tracklet_coverage.py` 추가. tracklet 17개 다운로드, GT와 같은 방식으로 투영해 평가 영역 안에서 `gt_depths.npz`와 422장 모두 일치 확인(영역 밖 97장 1–2픽셀 차이는 upstream `sub2ind` 인덱스 충돌). 점 10개 이상 물체 1,760개(차량 1,642, 보행자 37, 자전거 55, 기타 26; 0–10m 391개), 356장. 출력 `outputs/tracklets/`(coverage.md, objects.csv, labels.npz). 리포트 6.4절 하이라이트를 결과로 교체(하이라이트 0개), docx 재빌드. `EXPERIMENT_LOG.md` E13 추가 |
| 2026-11-28 | (로컬) 리포트 숫자 전수 점검: 표의 숫자(재현 결과, 영역별 오차, DP2 비교, 예비 실험 ep4·ep5 전 항목, 거리별 AbsRel, 경계 지표, 손실 진단, 파라미터·연산량 증가율, 학습 설정·split 수)는 원본 로그·CSV에서 다시 계산해 모두 일치. 숫자와 맞지 않거나 과한 서술을 고침: (1) 6.5절 "두 checkpoint 모두 범위 밖은 A-lite뿐" → A의 물체 놓침도 범위 밖(22.8%, 25.3%), baseline 물체 놓침도 ep10 이후 22.4–23.4%로 줄어든다는 점 추가 (2) 초록·6.6·6.7절 "약 8픽셀 이내 구분 못함" → 구분 못함(50%)은 바로 옆, 8px까지는 치우침 (3) 6.6절 "1.5배, 5.8 표준오차" → 원본 CSV 재계산 1.4배(34.8% vs 24.4%), 5.5 표준오차, 중간 대비 행 추가, "고대비에서 더 강함"은 하위와 상위 차이 1.5 표준오차라 삭제, 상관관계임을 명시 (4) 6.5·6.7절 번짐 "학습이 수렴하는 결과에 포함된 오차" → 학습 1회 결과로 한정 (5) 3장·5.3절 "decoder 구조 때문에" 단정 제거 (6) 5.4절·그림 4 캡션의 Lite-Mono 단독 실패 장면 서술을 그림과 맞게 고침 (7) 6.2·6.5절 smoothness 가중치를 재측정값으로 고침(`analysis/smoothness_weight_stats.py` 추가) (8) 6.4절 물체 AbsRel 비교를 같은 기준(같은 422장, 점 단위: 0.127 vs 0.072)으로 고치고 점 10개·0.15m 기준이 임의값임을 명시. `EXPERIMENT_LOG.md` 같은 내용 반영(해석 정정 기록 4번). docx 재빌드 |
| 2026-11-28 | (로컬) 임의 스크립트 결과 제외(사용자 요청: 원본 출력만 사용). 리포트에서 임의로 만든 코드로 계산한 값을 빼고 분석 스크립트의 원본 출력에 있는 값만 남김: (1) 초록·6.6·6.7절 손실 진단 1.4배·10.3%p·5.5 표준오차·대비별 표준오차 → `loss_preference.md`의 35% vs 24%(11%p), 대비별 32/24, 34/24, 40/25% (2) 6.2절 smoothness 150장 측정값 → 코드의 식(exp(−|∂I|) ≥ 0.37)으로만 설명, 6.5절 C 계수 근거의 측정값 삭제 (3) 6.4절 tracklet 커버리지 수치 전부 삭제(일부 주행에만 공개된다는 한계만 남김) (4) 5.4절 8장 장면 분류 삭제. `analysis/tracklet_coverage.py`, `analysis/smoothness_weight_stats.py`, `outputs/tracklets/`, `outputs/smoothness_weight_stats.md` 제거(받은 tracklet 원본 데이터는 유지). `EXPERIMENT_LOG.md` 해석 정정 기록 5번, E13 제외 처리. docx 재빌드 |
| 2026-11-28 | (로컬) 개선안 E(Depth Pro 2 decoder 구조) 구현: `networks/depth_decoder.py`에 `DPTDepthDecoder`, `options.py`·`trainer.py`·`evaluate_depth.py`·`analysis/baseline_failure_analysis.py`에 옵션 연결, `analysis/model_cost.py` 추가(출력 `outputs/model_cost.md`). 배치 1개 손실·역전파 확인만 했고 학습은 하지 않음. `vendor/changes_from_upstream.patch` 재생성. 리포트: 후보 표에 E 추가, 6.3절 출처 갱신, 6.7절 가설 1 판정을 "작은 변경으로는 확인되지 않음"으로 완화하고 종합 결론 수정, 6.8절에 E 계획 추가, 참고문헌 [14] DPT 추가. docx 재빌드. `EXPERIMENT_LOG.md` E14 추가 |
| 2026-11-28 | (로컬) `analysis/model_cost.py`에 추론 시간 측정 추가(batch 1, FP32, GPU RTX 5070 500회 중앙값·p90, CPU 4 thread 50회 중앙값). 출력 `outputs/model_cost.md`. Baseline GPU 3.59ms/CPU 51.8ms, E dim 24 +6%/+1%, E dim 64 +17%/+39%. 두 번 실행해 재현 확인. `EXPERIMENT_LOG.md` E14에 결과 추가. 리포트는 아직 수정하지 않음(채택 기준 변경은 사용자 결정 대기) |
| 2026-11-28 | (로컬) Lite-Mono 논문 원문(arXiv 2211.13202)과 공식 README의 속도 기록 확인. Table 4: 640×192, batch 16, Lite-Mono TITAN Xp 4.5ms, Jetson Xavier 20.0ms(이미지당). README 그래프의 batch 1은 9.1ms / 24.6ms이고, 같은 batch 1에서 Monodepth2는 7.1ms / 18.6ms. 학습 GPU(TITAN Xp, batch 12) 원문 확인 후 1장 표 갱신. `analysis/model_cost.py`에 논문과 같은 batch 16(이미지당) GPU 측정 추가: 로컬 baseline 2.02ms(batch 1은 3.59ms), E dim 24 +8%, E dim 64 +41%. `EXPERIMENT_LOG.md` E14 갱신 |
| 2026-11-28 | (로컬) 채택 기준 변경(사용자 결정): 연산량 증가 3% 이내 → baseline 대비 추론 시간 증가 10% 이내(GPU 1장씩, GPU 16장씩, CPU 1장씩 모두). 리포트: 초록 문구, 목표 문단, 후보 표(시간 열 추가, 캡션), 6.4절 기준·논문 속도 표 추가, 6.5절 표에 추론 시간 행 추가, 6.8절 E 시간 추가. docx 재빌드. 후보 E 5 epoch pilot 시작(`run_pilots.sh E` → `outputs/20261128_dpt24_e5`, 15:12). `run_pilots.sh`에 E 추가, `--dpt_*` 평가 인자 전달, ep4(weights_3)도 평가·분석하도록 변경. `baseline_failure_analysis.py`의 DPT 인자를 학습 옵션과 같은 `--dpt_decoder --dpt_dim --dpt_head_ch`로 통일. `pilot_summary.py`·`ad_metrics.py`·`pilot_status.py`에 E 추가 |
| 2026-11-28 | (로컬) 후보 E pilot 완료(16:22). `ad_metrics.py` 재실행으로 `pilots_ad.md`에 E 열 추가(기존 열은 그대로). 리포트: 6.5절 표에 E 열 추가와 결과 문단, 초록, 후보 표, 6.7절 가설 1 판정을 "5 epoch 예비 실험에서는 지지되지 않음"으로 바꾸고 종합 결론 수정, 6.8절 E 문단을 결과 기준으로 수정. docx 재빌드. `EXPERIMENT_LOG.md` E14 결과 추가 |
