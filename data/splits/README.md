# Splits

실제 사용하는 목록의 사본이다. 코드가 읽는 위치는 `vendor/Lite-Mono/splits/`이며 내용은 같다. 두 곳의 sha256이 다르면 어느 쪽이 바뀌었는지 먼저 확인한다.

| 파일 | 출처 | 행 수 | sha256 |
|---|---|---:|---|
| `train_files.txt` | Monodepth2 `splits/eigen_zhou/` | 39,810 | `00173a231642151c4693274e701ddd4cae7916520a757c7a94eae8e1173f7b0c` |
| `val_files.txt` | Monodepth2 `splits/eigen_zhou/` | 4,424 | `64a868b8d2c66c3979a2d74c22ea2e9146850d5b9f5786ed8e5f92201ba1ad04` |
| `test_files.txt` | Lite-Mono `splits/eigen/` (Monodepth2와 동일) | 697 | `e226d4ca16e0e6e9695435014347dcf973588531330536fd8a68d298ef091a6b` |

- Monodepth2 commit: `19e16e1c29e103a5a7c6e2f20dc9ac33210aa396`
- Lite-Mono commit: `4874b35df8ed4da16159ce8be8c697028b72bf76`
- 받은 날: 2026-11-08. 원본에서 수정한 행 없음.

Lite-Mono 원문은 학습 샘플을 39,180개로 적었지만 공개 목록은 39,810행이다. 원문 오타인지 필터링인지는 아직 확인하지 못했다.

train 목록의 날짜별 행 수: 2011_09_26 12,471 / 2011_09_28 161 / 2011_09_29 618 / 2011_09_30 16,153 / 2011_10_03 10,407.

split 목록으로 확인한 사실 (2026-11-08, 데이터 없이 목록만 분석):

- KITTI raw 다운로드 66개 = drive 61개 + calib 5개. 세 split이 쓰는 drive는 60개이고 `2011_09_26_drive_0060`은 어느 split에도 없다.
- drive 수: train 32 / val 31 / test 28. train과 test는 drive도 프레임도 겹치지 않는다. val의 31개 drive는 모두 train drive에 포함된다(장면 단위로 분리된 검증셋이 아님).
- 같은 행(drive, 프레임, 카메라)은 train과 val에 중복되지 않는다. 하지만 val 4,424행 중 3,950행은 train에 있는 **같은 프레임의 반대쪽 카메라** 영상이다. val로 checkpoint를 고르면 결과가 낙관적일 수 있다. 최종 수치는 test 697로 보고한다.
- 평가 LiDAR가 필요한 고유 프레임(val+test): 4,884개. 학습·평가에 필요한 고유 RGB(train/val ±1 + test): 45,259장.
- train 행의 카메라: 왼쪽(l) 19,956 / 오른쪽(r) 19,854.
- train+val 학습에 필요한 고유 RGB 이미지(각 행의 t−1, t, t+1, 해당 카메라): 44,562장.
- 코드 기본값은 `.jpg`를 읽는다(`trainer.py:152`, `--png` 미사용 시).

아직 하지 않은 점검: KITTI 파일 존재 여부, 인접 프레임(±1) 누락.

split을 임의로 바꾸거나 누락 파일을 조용히 제거하지 않는다.
