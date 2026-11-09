# 지금까지 바꾸고 추가한 것 (2026-11-08 ~ 11-09)

한 줄 요약: **논문의 모델·손실·데이터·하이퍼파라미터·평가 방식은 그대로다.** 바꾼 것은 (1) 최신 라이브러리에서 실행되게 하는 수정, (2) Colab에서 끊겨도 학습을 이어가게 하는 기능, (3) 새로 만든 준비·평가 도구뿐이다.

## 1. 가져온 것

| 무엇 | 어디서 | 위치 |
|---|---|---|
| Lite-Mono 공식 코드 | github.com/noahzn/Lite-Mono, commit `4874b35` (논문 abstract에 적힌 주소) | `vendor/Lite-Mono/` |
| 학습·검증 목록 (39,810 / 4,424) | Monodepth2 commit `19e16e1`. Lite-Mono 저장소에 빠져 있음 | `vendor/Lite-Mono/splits/eigen_zhou/`, 사본 `data/splits/` |
| 평가 정답 생성 코드 `export_gt_depth.py` | Monodepth2. Lite-Mono 저장소에 빠져 있음 | `vendor/Lite-Mono/` |
| 공식 가중치 (ImageNet 초기화, KITTI 학습 완료) | 공식 README의 SURFdrive 링크. sha256 기록 | `pretrained/` (Git 제외) |
| KITTI 다운로드 URL 66개 | Monodepth2 | `data/kitti_archives_to_download.txt` |

## 2. 공식 코드 수정 — 실행되게 하는 수정 6개

옛 라이브러리 함수가 지금은 사라져서, 그대로 두면 실행 도중 오류가 난다. 같은 동작을 하는 새 이름으로 바꿨다. 결과에 영향 없음. 코드에는 `# compat:` 주석.

| 파일 | 무엇이 문제였나 | 어떻게 고쳤나 |
|---|---|---|
| `datasets/mono_dataset.py` | 이미지 축소 방식 이름 `ANTIALIAS`가 Pillow 10에서 삭제 | 같은 방식의 현재 이름 `LANCZOS` |
| `networks/resnet_encoder.py` | PoseNet ImageNet 가중치 주소표 `model_urls`가 torchvision에서 삭제 | 같은 파일(`resnet18-f37072fd.pth`)을 가리키는 새 API |
| `trainer.py` (val) | `iterator.next()` 문법이 PyTorch에서 삭제 | `next(iterator)` |
| `trainer.py` (load_pretrain) | PyTorch 2.6부터 기본 보안 설정이 공식 ImageNet 가중치 파일을 거부 | 체크섬을 확인한 그 파일에만 `weights_only=False` |
| `kitti_utils.py` | `np.int`가 NumPy 1.24에서 삭제 | `int` (원래 같은 것) |
| `export_gt_depth.py` | 날짜마다 이미지 크기가 달라 최신 NumPy가 배열 생성을 거부 | 옛 NumPy가 자동으로 만들던 것과 같은 배열을 직접 생성 |

## 3. 공식 코드 수정 — Colab 운영용 기능 4개

Colab Pro 세션은 보통 12시간 안에 끊기고, Drive 용량도 한정되어 있어서 추가했다. 계산식은 그대로다. 코드에는 `# added:` 주석.

| 기능 | 하는 일 | 결과에 미치는 영향 |
|---|---|---|
| 이어서 학습 | 매 epoch 끝에 진행 상태(`train_state.json`)를 저장. 다시 실행하면 다음 epoch부터 같은 learning rate로 재개 | 계산은 같다. 다만 이어서 한 run은 데이터 순서가 끊김 없는 run과 달라서 완전히 똑같지는 않다 |
| `--seed` | B/I/C가 같은 초기값·데이터 순서·증강을 보게 함. 안 쓰면 원본처럼 seed 없음 | 공정한 비교용 |
| checkpoint 정리 | 매 epoch의 depth 모델은 모두 남기고, 재개에만 쓰는 큰 파일(약 175MB)은 최신 것만 남김. 메모리·누적 학습시간도 기록 | 없음 (저장 공간만 절약) |
| 진행률 표시 | 학습 로그 줄에 전체 진행률(%)을 넣고, 이어서 학습할 때 틀리던 남은 시간을 바로잡음. `time elapsed`는 세션을 합친 누적 시간 | 없음 (화면 출력만) |

전체 diff: `vendor/changes_from_upstream.patch` (6개 파일, +68/−16줄). 자세한 근거: `vendor/README.md`.

## 4. 새로 만든 파일 (공식 코드 밖)

| 파일 | 하는 일 |
|---|---|
| `scripts/download_kitti_pc.sh` | PC에서 아래 준비를 한 번에 실행하고 Drive에 올릴 파일 4개를 만듦 |
| `scripts/prepare_kitti.py` | KITTI zip을 하나씩 받아 필요한 프레임(45,259장)만 jpg로 변환하고 zip은 삭제. 평가용 LiDAR(val/test 4,884개)는 별도 폴더. 끝나면 빠진 파일 점검. 끊겨도 이어서 진행 |
| `analysis/eval_per_image.py` | 공식 평가와 같은 계산을 이미지별로 해서 CSV로 저장. val 4,424장도 메모리 부담 없이 평가하고 여러 checkpoint를 한 번에 평가 |
| `analysis/test_eval_per_image.py` | 위 평가 코드의 계산 점검 |
| `scripts/smoke_test.py` | 데이터 없이 Colab GPU에서 코드·가중치 일치, 파라미터 3.1M, 수정한 부분 동작 확인 |
| `scripts/colab_lite_mono.ipynb` | Colab 전체 순서: 설정 → 가중치 → 점검 → 데이터 복원 → 공개 checkpoint 평가(0.107 확인) → 학습 → checkpoint 선택 |
| `environment/requirements-colab.txt` | Colab에 추가로 설치할 패키지 |

## 5. 알아낸 사실 (보고서에 쓸 만한 것)

- 학습은 카메라 영상만 쓴다. LiDAR는 평가 정답을 만들 때만 쓴다.
- 논문은 학습 샘플을 39,180개로 적었지만 공개 목록은 39,810개다. val·test 수는 일치해서 오타로 보이지만 확인된 것은 아니다.
- train과 test는 장면이 겹치지 않는다. 하지만 val 4,424장 중 3,950장은 train에 있는 같은 순간을 반대쪽 카메라로 찍은 영상이다. val 점수는 낙관적일 수 있다.
- `options.py` 기본값(batch 16, 50 epoch)은 논문(batch 12, 30 epoch)과 다르다. 학습 명령에 직접 지정해야 한다.

## 6. 아직 확인 못 한 것

- 수정한 공식 코드를 실제로 실행해 보지 않았다. Colab C 셀(smoke test)과 F 셀(공개 checkpoint 0.107 재현)이 첫 검증이다.
- jpg 묶음과 정답 파일의 실제 크기, 실제 학습 시간은 데이터를 받은 뒤 측정한다.
