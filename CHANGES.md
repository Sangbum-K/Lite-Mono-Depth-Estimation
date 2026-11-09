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
| GPU | TITAN XP 1장 (논문 기재, 원문 재확인 필요) | RTX 5070 12GB | Colab 할당 GPU (세션마다 다름, `gpu_log.csv`에 기록) |
| Driver / CUDA | 기재 없음 | 580.178.04 / CUDA 12.8 | 세션마다 다름 |
| OS | 기재 없음 | Ubuntu 24.04.4 LTS | Colab 기본 이미지 |
| CPU / RAM | 기재 없음 | 16 threads / 15GB | Colab 할당 |
| Python | 기재 없음 (monodepth2 계열은 3.6~3.8대) | 3.11.15 (`.venv`) | Colab 기본 |
| PyTorch / torchvision | 구버전 (정확한 버전 기재 없음) | 2.11.0+cu128 / 0.26.0+cu128 | Colab 기본 (pip freeze로 기록) |
| NumPy / Pillow | 구버전 | 1.26.4 / 12.2.0 | Colab 기본 |
| 기타 패키지 | README의 pip 목록 | timm 1.0.31, tensorboardX 2.6.5, thop 0.1.1, scikit-image 0.26.0, opencv-python-headless 4.11.0.86 | `environment/requirements-colab.txt` (버전 미고정) |
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
| `export_gt_depth.py` | 최신 NumPy가 크기 다른 배열 묶음 거부 | 파일 없음 | 수정해서 추가 |

로컬 수정 내역 전체: `vendor/changes_from_upstream.patch` (upstream `4874b35` 대비 4개 파일 +19/−8줄). 중첩 `.git`은 저장소 밖 `/home/user/sangbum/Lite-Mono-upstream.git`으로 옮겼다.

### 4.2 기능 추가

| 기능 | 로컬 | Colab | 결과 영향 |
|---|---|---|---|
| `--seed` | 없음 | 있음 | ➖ 안 쓰면 원본과 같음 |
| 이어서 학습 (`train_state.json`) | 없음 | 있음 | ➖ 계산은 같음. 이어서 한 run은 데이터 순서가 끊김 없는 run과 달라짐 |
| checkpoint 정리 | 없음 (모든 epoch 전부 저장) | pose/optimizer 파일은 최신 것만 저장 | ➖ |
| 진행률·남은 시간 출력 | 없음 | 있음 | ➖ |
| pretrain 로드 메시지 | 불러온 텐서 수 출력 (`233/263`) | 원본 그대로 | ➖ |

### 4.3 upstream에 없어서 추가한 파일 (monodepth2 commit `19e16e1`)
- `splits/eigen_zhou/{train,val}_files.txt`: 로컬 ✅ / Colab ✅
- `export_gt_depth.py`, `LICENSE_monodepth2`: 로컬 ✗ / Colab ✅

## 5. 실행 기록

| run | 환경 | 날짜 | 조건 | 논문과 다른 점 | 상태 |
|---|---|---|---|---|---|
| `20261109_baseline_unseeded` | 로컬 | 2026-11-09 시작 | 공식 하이퍼파라미터, ImageNet 초기화 | ⚠️ PNG 입력, seed 없음, num_workers 6 | 진행 중 (2026-11-09 기준 epoch 12/30, `weights_0`~`11` 저장됨) |

- 이 run은 PNG 입력이라 논문 수치와 직접 비교하는 baseline으로 쓰기 어렵다. 공식 코드를 그대로 돌린 참고용 결과로 다룬다.

## 6. 알려진 문제

- ⚠️ **Colab 브랜치에 `vendor/Lite-Mono/datasets/` 폴더가 빠져 있다.** 루트 `.gitignore`의 `datasets/` 규칙(KITTI 데이터 제외용)이 코드 폴더까지 막았다. 이대로면 Colab에서 clone 후 `train.py`가 `import datasets`에서 실패한다. 또 CHANGES에 적힌 `mono_dataset.py`의 ANTIALIAS 수정도 실제로는 커밋되지 않았다. → `.gitignore`를 `/datasets/`처럼 루트 기준으로 좁히거나 `!vendor/Lite-Mono/datasets/`를 추가한 뒤 커밋해야 한다.
- `Local` 브랜치에서는 `.gitignore`를 `/datasets/`로 고쳐 해결했다. Colab 브랜치에는 아직 반영되지 않았다.

## 7. 아직 확인 못 한 것

- 공개 checkpoint(`pretrained/kitti_640x192`)로 논문 AbsRel 0.107이 재현되는지 (로컬·Colab 모두 미확인).
- 논문의 학습 GPU·소프트웨어 버전 원문 표기.
- PNG와 JPG 학습의 실제 성능 차이.

---

## 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-11-08 ~ 09 | (Colab 브랜치) 공식 코드 반입, 호환 수정, Colab 이어서 학습·seed 기능, JPG 준비 스크립트 |
| 2026-11-09 | (로컬) 공식 코드 반입·호환 수정, PNG·unseeded baseline 학습 시작 |
| 2026-11-09 | 로컬·Colab을 함께 다루는 CHANGES.md 작성. Colab 브랜치 `datasets/` 누락 발견 |
| 2026-11-09 | 로컬 코드를 `Local` 브랜치에 커밋. `.gitignore` `datasets/` → `/datasets/`, 로컬 수정 patch 저장 |
