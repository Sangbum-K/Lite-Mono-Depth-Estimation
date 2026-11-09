# Vendor code

`vendor/Lite-Mono/`는 Lite-Mono 공식 코드의 복사본이다. 받은 날: 2026-11-08.

| 항목 | 값 |
|---|---|
| upstream | https://github.com/noahzn/Lite-Mono |
| commit | `4874b35df8ed4da16159ce8be8c697028b72bf76` (2023-12-11, "update pre-training and instructions") |
| 라이선스 | MIT, `vendor/Lite-Mono/license` |

중첩 `.git`은 지웠다. 상위 저장소에 올릴 때 submodule처럼 코드가 빠지는 것을 막기 위해서다. upstream 원본과의 차이는 위 commit을 다시 clone해서 `diff -r`로 확인한다.

## upstream에 없어서 추가한 파일

Lite-Mono 코드가 읽지만 저장소에 포함되지 않은 파일이다. 모두 Monodepth2에서 그대로 복사했고 내용은 수정하지 않았다.

- 출처: https://github.com/nianticlabs/monodepth2 commit `19e16e1c29e103a5a7c6e2f20dc9ac33210aa396`
- 라이선스: Niantic 비상업 라이선스, `vendor/Lite-Mono/LICENSE_monodepth2`

| 파일 | 필요한 이유 |
|---|---|
| `splits/eigen_zhou/train_files.txt` | `trainer.py:148`이 기본 split `eigen_zhou`의 목록을 읽음 |
| `splits/eigen_zhou/val_files.txt` | 위와 같음 |
| `export_gt_depth.py` | `evaluate_depth.py:166`이 읽는 `splits/eigen/gt_depths.npz`를 KITTI velodyne에서 생성. Lite-Mono의 `utils.readlines`, `kitti_utils.generate_depth_map`을 그대로 import함 (`kitti_utils.py`는 두 저장소가 빈 줄 하나 외에 동일) |

`splits/eigen/test_files.txt`는 Lite-Mono에 원래 있고 Monodepth2의 파일과 동일하다.

## 가중치

`pretrained/`에 두며 Git에서 제외된다. README 표의 SURFdrive 링크에서 받았다.

| 파일 | 용도 | SURFdrive | sha256 |
|---|---|---|---|
| `pretrained/lite-mono-pretrain.pth` | ImageNet encoder 초기화, 재학습용 `--mypretrain` | `InMMGd5ZP2fXuia` | `9877014dfaede372ca887dea354cb534788eb68b6ded13a9a1d39c715c04fc9f` |
| `pretrained/lite-mono_640x192/encoder.pth` | KITTI 학습 완료 모델, 평가 점검용 | `CUjiK221EFLyXDY` | `9c5bc031b4a6a8d9c6df5f525c6740594b7d814841cc924f61565d0da624e046` |
| `pretrained/lite-mono_640x192/depth.pth` | 위와 같음 | `CUjiK221EFLyXDY` | `c452dd45e23a4cddd823943a43e793537c0095c0c8fc2037a8a4e6ad30acc224` |

## 수정 기록

모든 코드 변경은 [changes_from_upstream.patch](changes_from_upstream.patch)에 있다 (6개 파일, +59/−10줄). 코드 안에서는 `# compat:`(환경 호환)과 `# added:`(Colab 운영) 주석으로 표시했다. **모델 구조·손실·데이터·하이퍼파라미터·평가 방식은 바꾸지 않았다.** 연구 아이디어 변경은 아직 없으며, 생기면 별도 표로 추가한다.

### 환경 호환 수정 (2026-11-08)

최신 라이브러리(torch 2.14, torchvision 0.29, Pillow 12, NumPy ≥1.24)의 소스에서 제거·변경을 확인했다. 고치지 않으면 실행 중 오류가 난다. 모두 원래와 같은 동작이다.

| 위치 | 원래 코드 | 수정 | 같은 동작인 이유 |
|---|---|---|---|
| `datasets/mono_dataset.py:49` | `Image.ANTIALIAS` | `Image.LANCZOS` | Pillow에서 ANTIALIAS는 LANCZOS의 별칭이었음 |
| `networks/resnet_encoder.py:48` | `models.resnet.model_urls[...]` | `get_model_weights(...).IMAGENET1K_V1.url` | 같은 파일 `resnet18-f37072fd.pth` |
| `trainer.py` `val()` | `self.val_iter.next()` | `next(self.val_iter)` | 같은 iterator 호출 |
| `trainer.py` `load_pretrain()` | `torch.load(path)` | `weights_only=False` | torch 2.6부터 기본값이 바뀌어 `argparse.Namespace`가 든 공식 파일을 거부. 체크섬 확인한 공식 파일에만 사용 |
| `kitti_utils.py:85` | `astype(np.int)` | `astype(int)` | `np.int`는 `int`의 별칭이었음 |
| `export_gt_depth.py:61` (Monodepth2 사본) | `np.array(gt_depths)` | object 배열을 명시적으로 생성 | 날짜별로 이미지 크기가 달라 옛 NumPy가 자동으로 만들던 것과 같은 1차원 object 배열 |

`networks/resnet_encoder.py`의 `resnets[num_layers](pretrained)` 줄도 최신 torchvision에서는 동작하지 않지만, PoseNet은 두 장을 입력으로 받는 다른 경로를 써서 이 줄을 실행하지 않으므로 두었다. `timm.models.layers` import는 경고만 낸다.

### Colab 운영을 위한 추가 (2026-11-08)

| 위치 | 내용 | 학습 결과에 미치는 영향 |
|---|---|---|
| `options.py`, `trainer.py` | `--seed`. 지정하지 않으면 원본처럼 seed 없음 | 지정하면 B/I/C가 같은 초기화·데이터 순서·증강을 봄 |
| `trainer.py` | 이어서 학습: `train_state.json`(epoch, step, 누적 시간, peak GPU memory)을 저장하고, `--load_weights_folder`에 이 파일이 있으면 다음 epoch부터 재개. scheduler는 epoch 수만큼 `step()`을 다시 불러 같은 learning rate를 복원. 재개할 때는 `--mypretrain`으로 덮어쓰지 않음 | 계산식은 같음. 재개한 run은 끊김 없는 run과 데이터 순서가 달라 비트 단위로 같지는 않음 |
| `trainer.py` `save_model()` | 매 epoch의 encoder/depth는 모두 보관하고, 재개에만 필요한 pose·optimizer 파일(~175MB)은 최신 epoch 것만 남김 | 없음 (저장 공간만 절약) |

## 가중치와 코드 일치 확인 (2026-11-08, 정적 검사)

가중치 파일을 실행하지 않고 key 이름과 텐서 크기만 읽었다. 코드에서 모델을 만들어 실제로 로드하는 검사는 이 PC에서 외부 코드 실행이 막혀 Colab에서 해야 한다.

- KITTI encoder: stage별 block 4/4/10, 각 stage 마지막 block이 LGFI. `networks/depth_encoder.py`의 `lite-mono` 설정(depth `[4,4,10]`, `global_block [1,1,1]`)과 일치. 저장된 `height=192, width=640, use_stereo=False`.
- KITTI decoder: 텐서 18개 = ConvBlock 6개 + dispconv 3개(scale 0–2). `scales=range(3)` 코드와 일치.
- encoder+decoder 저장값 합계 3,078,637개(BatchNorm 통계 포함) ≈ 논문의 3.1M.
- 코드에 없는 key: encoder의 `norm.weight/bias`, thop 측정용 `total_ops/total_params`. 공식 로더는 모델에 있는 key만 골라 로드하므로 무시된다.
- ImageNet pretrain: 같은 4/4/10 구조. 분류용 `head.*`, `norm.*`는 `trainer.load_pretrain()`에서 제외된다.

최종 확인은 공개 checkpoint를 공식 `evaluate_depth.py`로 평가해 README의 AbsRel 0.107이 나오는지로 한다.
