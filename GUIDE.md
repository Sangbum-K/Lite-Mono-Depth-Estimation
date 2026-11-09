# 전체 안내서: 무엇이 어디 있고, Colab에서 무엇이 어떻게 돌아가나

이 문서 하나로 프로젝트 전체를 이해할 수 있게 정리했다. 처음 읽는다면 1~4절만 읽어도 Colab을 돌릴 수 있다. 5절은 결과 숫자 읽는 법, 6절은 파일별 사전, 8절은 오류가 났을 때 본다.

---

## 1. 이 프로젝트가 하는 일

**사진 한 장을 넣으면 픽셀마다 거리(깊이)를 내놓는 모델**을 학습한다.

- 모델: **Lite-Mono** (CVPR 2023). 파라미터 약 3.1M개의 작은 깊이 추정 모델이다.
- 데이터: **KITTI**. 독일에서 차에 카메라와 LiDAR(레이저 거리 센서)를 달고 주행하며 찍은 데이터다.
- 순서: ① 논문 결과를 직접 재현(baseline)한다. ② 어디서 틀리는지 분석한다. ③ decoder를 조금 고쳐 같은 조건으로 다시 학습하고 비교한다.

학습에는 **사진만** 쓴다. 정답 깊이 없이, 연속된 사진끼리 비교해서 배운다(자기지도 학습, 4절). LiDAR는 채점할 때 정답으로만 쓴다.

---

## 2. 전체 그림: PC → Drive → Colab

```
 ┌───────────── PC (이미 끝남) ─────────────┐
 │ scripts/download_kitti_pc.sh             │
 │  KITTI zip 60개(173GB)를 하나씩 받아서     │
 │  필요한 것만 남기고 zip은 삭제             │
 │        ↓                                 │
 │ data/processed/  파일 4개 (7.2GB)         │
 └──────────────────┬───────────────────────┘
                    │ 손으로 업로드
                    ↓
 ┌────── Google Drive  내 드라이브/lite_mono/ (영구 보관) ──────┐
 │ data/        사진 묶음 + 정답 깊이 3개   ← PC에서 올린 것      │
 │ pretrained/  저자 공개 가중치            ← 노트북 2단계가 받음  │
 │ env/         세션별 GPU·패키지 기록      ← 노트북 1단계가 씀    │
 │ outputs/     학습 결과(epoch별 모델, 로그) ← 노트북 6단계가 씀   │
 └──────────────────┬──────────────────────────────────────────┘
                    │ 매 세션 연결
                    ↓
 ┌────── Colab 세션 /content/ (세션이 끝나면 지워짐) ──────┐
 │ Lite-Mono-Depth-Estimation/  코드 (GitHub Colab 브랜치) │
 │ kitti_data/                  Drive tar를 푼 사진 45,259장 │
 │ GPU에서 학습·채점 → 결과는 Drive에 저장                    │
 └─────────────────────────────────────────────────────────┘
```

| 장소 | 무엇이 있나 | 수명 |
|---|---|---|
| **PC** `~/Lite-Mono-Depth-Estimation-main` | 코드 원본(여기서 고쳐서 GitHub에 push), KITTI 작업 폴더 | 영구 |
| **GitHub** `Colab` 브랜치 | Colab이 받아 가는 코드 | 영구 |
| **Drive** `lite_mono/` | 데이터, 가중치, 학습 결과 | 영구 |
| **Colab** `/content/` | 코드 사본, 압축을 푼 사진 | 세션이 끝나면 사라짐. 그래서 노트북 1·2·4단계를 매 세션 다시 실행 |

**코드를 고치는 흐름:** PC에서 수정 → GitHub `Colab` 브랜치에 push → Colab 1단계가 최신 코드를 받아 옴(이미 받아 둔 코드는 자동 업데이트).

---

## 3. Colab 노트북 단계별 설명

파일: [scripts/colab_lite_mono.ipynb](scripts/colab_lite_mono.ipynb)

| 상황 | 실행할 단계 |
|---|---|
| 맨 처음 | 1 → 2 → 3 → 4 → 5 |
| 학습 시작, 또는 끊긴 학습 이어하기 | 1 → 2 → 4 → 6 |
| 학습이 끝난 뒤 | 1 → 2 → 4 → 7 |

### 1단계. 준비 (매 세션)

| | |
|---|---|
| 하는 일 | ① Drive 연결 ② 경로와 학습 이름(`RUN_ID`) 설정 ③ Drive에 데이터 4개가 있는지 확인 ④ GitHub에서 코드 받기(있으면 최신으로 업데이트) ⑤ 추가 패키지 설치 ⑥ GPU·패키지 버전을 Drive `env/`에 기록 |
| 쓰는 파일 | `environment/requirements-colab.txt` |
| 정상 출력 | `코드 버전: <커밋 번호> <커밋 메시지>`, GPU 이름(예: `Tesla T4`), `준비 완료. 학습 이름: ...` |
| 주의 | 로그인 팝업에서 **Drive와 같은 계정**, **권한 체크박스 전부 선택**. `RUN_ID`는 한 학습이 끝날 때까지 바꾸지 않는다 |

설정값:

| 이름 | 값 | 의미 |
|---|---|---|
| `DRIVE` | `/content/drive/MyDrive/lite_mono` | Drive의 프로젝트 폴더 |
| `REPO` | `/content/Lite-Mono-Depth-Estimation` | Colab에 받은 코드 |
| `KITTI` | `/content/kitti_data` | 압축을 푼 사진 |
| `RUN_ID` | `20261109_baseline_s42` | 학습 이름 = 결과 폴더 이름. `날짜_방법_seed` |
| `SEED` | 42 | 난수 고정값. 비교하는 학습끼리 같게 둔다 |
| `NUM_WORKERS` | 8 | 사진을 읽는 병렬 프로세스 수. 데이터 증강 난수에 영향을 주므로 비교하는 학습끼리 같게 둔다 |

### 2단계. 공식 가중치 (매 세션)

| | |
|---|---|
| 하는 일 | 저자가 공개한 가중치를 Drive `pretrained/`에 받는다(처음 한 번). 그다음부터는 sha256 체크섬으로 파일이 손상되지 않았는지만 확인한다 |
| 받는 것 | `lite-mono-pretrain.pth`: ImageNet으로 미리 학습한 인코더. **우리 학습의 출발점**<br>`lite-mono_640x192/encoder.pth`, `depth.pth`: 저자가 KITTI 학습을 끝낸 모델. **5단계 점검용** |
| 정상 출력 | 세 줄 모두 `OK` |

### 3단계. 코드 점검 smoke test (처음 한 번, 코드를 바꾼 뒤)

| | |
|---|---|
| 하는 일 | 데이터 없이 몇 초 만에 코드가 돌아가는지 확인 |
| 쓰는 파일 | `scripts/smoke_test.py` |
| 정상 출력 | 마지막 줄 `smoke test passed` (2026-11-09 통과) |

확인 항목:
1. 우리 코드 구조에 저자 모델이 빠짐없이 들어감. 파라미터 3,074,747개(논문 3.1M)
2. ImageNet 인코더 가중치가 학습 코드와 같은 방식으로 들어감(263개 중 233개. 나머지 30개는 사전학습 파일에 없는 LayerNorm이라 기본값 1/0으로 시작하고, 공식 학습도 같다)
3. PoseNet(ResNet18) ImageNet 가중치 다운로드
4. 데이터 로더가 가짜 사진 3장으로 입력 하나를 만듦(`depth_gt loaded: False`가 정상. 학습에는 LiDAR가 없다)
5. GPU에서 batch 12로 forward와 backward 한 번. 깊이를 3개 해상도(192×640, 96×320, 48×160)로 출력

### 4단계. 데이터 풀기 (매 세션)

| | |
|---|---|
| 하는 일 | ① Drive의 `kitti_jpg.tar`(6.3GB)를 `/content/kitti_data`에 푼다 ② `eigen_gt_depths.npz`를 공식 채점 코드가 읽는 위치(`vendor/Lite-Mono/splits/eigen/gt_depths.npz`)에 복사 ③ 필요한 사진이 전부 있는지 점검 |
| 왜 | Drive에서 작은 파일 수만 개를 바로 읽으면 매우 느리다. 큰 파일 하나를 가져와 로컬 디스크에 푸는 게 빠르다 |
| 쓰는 파일 | `scripts/prepare_kitti.py` (점검만 하고 다운로드는 안 함. tar 안의 `.done` 표시 덕분) |
| 정상 출력 | `need 45259 RGB frames, 0 LiDAR frames, 60 drives` 다음 `check: RGB 45259/45259 present, LiDAR 0/0 present, calib missing 0` |

### 5단계. 저자 모델 채점 (처음 한 번)

| | |
|---|---|
| 하는 일 | 저자가 학습을 끝낸 모델로 test 697장을 채점해서, **데이터 준비와 채점 코드가 맞는지** 확인 |
| 쓰는 파일 | `vendor/Lite-Mono/evaluate_depth.py`(공식 채점), `analysis/eval_per_image.py`(이미지별 채점) |
| 정상 출력 | 두 결과 모두 `0.107 0.765 4.561 0.183 0.886 0.963 0.983` 근처 |
| 다르면 | 학습하지 말고 원인부터 찾는다. 데이터나 채점이 틀렸다면 이후 숫자를 모두 믿을 수 없다 |

두 채점 코드의 차이: 공식 코드는 전체 평균만 출력한다. 이미지별 코드는 같은 계산을 하면서 이미지마다 점수를 CSV(`per_image_test_files.csv`)로 남긴다. 실패 분석에 필요하다. 두 평균이 같으면 이미지별 코드도 맞다는 증거가 된다.

### 6단계. 학습 (끊기면 1 → 2 → 4 → 6으로 이어서)

| | |
|---|---|
| 하는 일 | Lite-Mono를 논문 설정 그대로 30 epoch 학습 (원리는 4절) |
| 쓰는 파일 | `vendor/Lite-Mono/train.py` → `trainer.py`, `options.py`, `networks/`, `datasets/`, `layers.py` |
| 처음 실행 | ImageNet 인코더(`lite-mono-pretrain.pth`)에서 시작 |
| 다시 실행 | Drive `outputs/<RUN_ID>/models/`에서 `train_state.json`이 있는 마지막 epoch을 찾아 그다음 epoch부터 이어간다 |
| 결과 | Drive `outputs/<RUN_ID>/` (7절) |

- **세션 유지**: 학습 직전에 노트북 6단계 설명에 있는 KeepAlive 코드를 브라우저 콘솔(F12)에 붙여넣는다. 유휴 상태로 판정돼 끊기는 것을 막는다. 12시간 제한은 늘어나지 않는다.
- **TensorBoard**(선택): 학습 셀보다 먼저 실행해 두면 학습 중 loss 그래프와 예측 깊이 이미지를 볼 수 있다. 학습 셀이 도는 동안에는 다른 셀을 실행할 수 없기 때문이다.

### 7단계. epoch 고르기 → 최종 test 채점 (학습이 끝난 뒤)

| | |
|---|---|
| 7-1 | 30개 epoch 모델을 모두 val 4,424장으로 채점. 결과는 각 `weights_N/per_image_val_files.csv`, 요약은 `outputs/<RUN_ID>/val_summary.txt` |
| 7-2 | 미리 정한 규칙(예: val abs_rel 최저)으로 고른 epoch를 `CHOSEN_EPOCH`에 적고 test 697장으로 채점. **이 숫자가 보고서의 최종 결과** |
| 규칙 | test 점수를 보고 고르면 반칙이다. 고르는 규칙은 결과를 보기 전에 PLAN.md에 적어 둔다 |
| 주의 | val 4,424장 중 3,950장은 train에 있는 같은 순간을 반대쪽 카메라로 찍은 사진이다. 그래서 val 점수는 실제보다 좋게 나올 수 있다 |

---

## 4. 학습은 어떻게 돌아가나 (쉽게)

### 4.1 등장인물: 네트워크 두 개

| | 입력 → 출력 | 역할 | 학습 후 |
|---|---|---|---|
| **DepthNet** (= Lite-Mono) | 사진 1장 → 픽셀마다 깊이 | 우리가 만들려는 모델 | **남겨서 씀** (`encoder.pth` + `depth.pth`) |
| **PoseNet** (ResNet18) | 연속 사진 2장 → 그 사이에 카메라가 움직인 양(회전 3 + 이동 3) | 학습을 돕는 조수 | 버림 |

Lite-Mono 논문의 기여는 **DepthNet을 작고 정확하게 만든 설계**다. 학습 방식은 Monodepth2(2019)를 그대로 따른다.
- 인코더(`networks/depth_encoder.py`): CDC(칸 간격을 넓힌 합성곱을 연속으로 쌓아 넓은 범위를 봄)와 LGFI(사진 전체의 관계를 보는 가벼운 attention)를 섞었다.
- 디코더(`networks/depth_decoder.py`): 인코더 특징을 키우면서 3개 해상도로 깊이를 출력한다. **우리 프로젝트가 개선하려는 부분**이다.

### 4.2 정답 없이 어떻게 배우나: 한 스텝

```
연속 사진 3장 (t-1, t, t+1). 같은 카메라.
  │
  ├─ DepthNet(t)          → t의 깊이
  ├─ PoseNet(t, t-1)      → t에서 t-1까지 카메라 이동
  ├─ PoseNet(t, t+1)      → t에서 t+1까지 카메라 이동
  │
  ├─ 깊이 + 이동 + 카메라 보정값으로 t-1, t+1 사진을 t 시점에서 본 모습으로 "다시 그림"
  ├─ 다시 그린 사진 vs 진짜 t 사진 → 얼마나 다른가 = 손실(loss)
  └─ 손실이 줄어드는 방향으로 두 네트워크를 조금씩 수정 (AdamW)
```

깊이와 이동을 둘 다 맞혀야 다시 그린 사진이 진짜와 겹친다. 그래서 사진 비교만으로 깊이를 배운다.

손실의 세부 사항(`trainer.py`의 `compute_losses`):
- **사진 차이** = 0.85 × SSIM(구조 차이) + 0.15 × L1(밝기 차이)
- **최소값 선택**: t-1과 t+1 중 더 잘 맞는 쪽만 쓴다. 한쪽에서 가려진 부분 때문에 생기는 오류를 줄인다.
- **자동 마스킹**: 우리 차와 같이 움직이는 차, 정지 장면처럼 "움직였다고 볼 수 없는" 픽셀은 손실에서 뺀다.
- **부드러움**(가중치 0.001): 사진에 경계가 없는 곳에서는 깊이도 매끄럽게 변하도록 한다.
- 이 계산을 해상도 3단계에서 모두 한다.

**데이터 증강**: 사진마다 50% 확률로 색(밝기·대비·채도·색조)을 흔들고, 50% 확률로 좌우를 뒤집는다. 같은 사진도 매번 조금 다르게 보여 과적합을 막는다.

### 4.3 숫자로 본 학습 규모

| 항목 | 값 |
|---|---|
| 학습 샘플 | 39,810개 (eigen_zhou 목록. 한 줄 = 주행 폴더, 프레임 번호, 왼쪽/오른쪽 카메라) |
| batch | 12 (한 스텝에 샘플 12개) |
| 1 epoch | 3,317 스텝 (39,810 ÷ 12, 나머지 버림) = 모든 샘플을 한 번씩 봄 |
| 전체 | 30 epoch = 99,510 스텝 |
| 입력 크기 | 640×192 (원본 약 1242×375를 줄임) |
| 학습률 | DepthNet 0.0001에서 시작해 cosine 곡선으로 0.000005까지 서서히 감소. PoseNet은 0.0001 → 0.00001. 주기가 31 epoch라 30 epoch 동안 한 번 내려가고 끝난다 (`--lr 0.0001 5e-6 31 0.0001 1e-5 31`) |
| 학습 시간 | 아직 측정 전. 첫 학습의 `time left`와 `train_state.json`으로 확인 |

### 4.4 학습 중 화면 읽기

처음 약 20,000 스텝(약 6 epoch)은 250 스텝마다, 그 뒤로는 2,000 스텝마다 로그가 한 줄 나온다.

```
epoch   3 |  12.5% | lr 0.000097 |lr_p 0.000097 | batch   1500 | examples/s:  95.0 | loss: 0.12000 | time elapsed: 01h30m00s | time left: 10h30m00s
```
(숫자는 예시)

| 항목 | 의미 |
|---|---|
| `epoch 3` | 지금 몇 번째 epoch인지 (0부터 셈, 마지막은 29) |
| `12.5%` | 전체 30 epoch 중 진행률 |
| `lr`, `lr_p` | DepthNet, PoseNet의 현재 학습률. 점점 줄어든다 |
| `batch 1500` | 이번 epoch 안에서 몇 번째 스텝인지 |
| `examples/s` | 초당 처리한 샘플 수. GPU 속도 지표 |
| `loss` | 이번 스텝의 손실. 배치마다 흔들리므로 **전체적으로 내려가는지만** 본다 |
| `time elapsed` | 지금까지 학습한 시간. 끊겼다 이어도 합산된다 |
| `time left` | 남은 예상 시간 |

로그를 찍을 때마다 val 사진 한 묶음으로도 손실을 계산해 TensorBoard에 기록한다(`val/` 폴더). 이건 학습 상태를 보는 참고용이다. checkpoint 선택은 7단계에서 val 전체로 한다.

### 4.5 저장과 이어하기

epoch 하나가 끝날 때마다 Drive `outputs/<RUN_ID>/models/weights_<epoch>/`에 저장한다.

| 파일 | 내용 | 보관 |
|---|---|---|
| `encoder.pth`, `depth.pth` | DepthNet. 7단계 채점에 씀 | **모든 epoch** |
| `pose_encoder.pth`, `pose.pth` | PoseNet | 마지막 epoch만 (이어하기용) |
| `adam.pth`, `adam_pose.pth` | 옵티마이저 상태 | 마지막 epoch만 (이어하기용) |
| `train_state.json` | 몇 epoch·스텝까지 했는지, 누적 시간, 최대 GPU 메모리 | 마지막 epoch만 |

세션이 끊기면 1 → 2 → 4 → 6을 다시 실행한다. 6단계가 `train_state.json`이 있는 마지막 epoch 폴더를 찾아 모델·옵티마이저·학습률을 복원하고 다음 epoch부터 시작한다. **끊긴 epoch 도중의 진행분은 버려진다**(epoch 단위 저장). 이어서 한 학습은 끊김 없이 한 학습과 데이터 순서가 달라서 결과가 비트 단위로 똑같지는 않다.

---

## 5. 점수 읽는 법

채점은 이렇게 한다.
1. 모델이 test 사진의 깊이를 예측한다.
2. 사진 한 장으로는 실제 크기(미터)를 알 수 없다. 그래서 이미지마다 예측의 중앙값을 LiDAR 정답의 중앙값에 맞춘다(**median scaling**).
3. LiDAR 점이 있는 픽셀만, 이미지 아래쪽 약 60%의 정해진 영역(Eigen crop. LiDAR가 닿지 않는 하늘 쪽을 뺀다)에서, 80m까지만 비교한다.

| 지표 | 뜻 | 좋은 방향 | 저자 모델 |
|---|---|---|---|
| `abs_rel` | 평균 상대 오차. 0.107이면 평균 10.7% 틀림. **주 지표** | 낮을수록 | 0.107 |
| `sq_rel` | 제곱 상대 오차. 크게 틀린 픽셀에 민감 | 낮을수록 | 0.765 |
| `rmse` | 평균 제곱근 오차(미터) | 낮을수록 | 4.561 |
| `rmse_log` | 로그 깊이의 rmse | 낮을수록 | 0.183 |
| `a1` | 예측이 정답의 1.25배 이내인 픽셀 비율 | 높을수록 | 0.886 |
| `a2` | 1.25² = 1.56배 이내 비율 | 높을수록 | 0.963 |
| `a3` | 1.25³ = 1.95배 이내 비율 | 높을수록 | 0.983 |

공식 채점 출력 예:
```
   Mono evaluation - using median scaling
 Scaling ratios | med: ... | std: ...
   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 |
&   0.107  &   0.765  &   4.561  &   0.183  &   0.886  &   0.963  &   0.983  \\
```

---

## 6. 파일별 설명

표시: **[Colab]** = 노트북이 실제로 실행하거나 읽는 파일, **[PC]** = PC에서만 쓴 파일, **[문서]** = 사람이 읽는 기록.

### 6.1 맨 위 폴더

| 파일 | 설명 |
|---|---|
| `README.md` [문서] | 프로젝트 소개, 마감일, 지금 할 일 |
| `GUIDE.md` [문서] | 이 문서 |
| `PLAN.md` [문서] | Phase 0~8 전체 계획, 각 Phase의 완료 기준, 실험 규칙(B/I/C 비교, checkpoint 선택 규칙 등) |
| `CHANGES.md` [문서] | 공식 코드에서 바꾼 것과 새로 만든 것의 쉬운 요약 |
| `STRUCTURE.md` [문서] | 폴더 구조 원칙, run ID 규칙(`날짜_방법_seed`), 백업 기준 |
| `HW2.pdf` [문서] | 과제 안내 |
| `.gitignore` | Git에 올리지 않을 것: KITTI 데이터, 가중치(`*.pth`), 정답(`*.npz`), 결과 폴더 등 |

### 6.2 `scripts/` — 실행 스크립트

| 파일 | 설명 |
|---|---|
| `colab_lite_mono.ipynb` [Colab] | Colab 노트북. 1~7단계 (3절) |
| `smoke_test.py` [Colab] | 3단계. 데이터 없이 코드·가중치·GPU 점검 |
| `prepare_kitti.py` [PC·Colab] | KITTI 준비 도구. PC에서는 zip 60개를 하나씩 받아 필요한 프레임만 jpg로 바꾸고(45,259장, 품질 92) zip은 지웠다. val/test LiDAR(4,884개)는 학습 폴더와 다른 폴더에 따로 저장했다. 받은 zip마다 `.done` 표시를 남겨 끊겨도 이어서 진행한다. Colab 4단계에서는 `--kitti_out`만 주어 **빠진 사진 점검**에만 쓴다 |
| `download_kitti_pc.sh` [PC] | PC에서 데이터 준비 전체를 한 번에 실행: `prepare_kitti.py` → 정답 깊이 3개 생성 → 사진을 tar로 묶음. 결과는 `data/processed/`의 4개 파일 |
| `README.md` [문서] | 위 파일들의 역할과 검증 상태 |

### 6.3 `analysis/` — 채점·분석

| 파일 | 설명 |
|---|---|
| `eval_per_image.py` [PC·Colab] | 공식 채점과 같은 계산을 **이미지별로** 한다. 명령이 두 개다. `build-gt`: LiDAR로 정답 npz를 만든다(PC에서 실행해 `test_gt.npz`, `val_gt.npz` 생성). `eval`: 모델 여러 개를 한 번에 채점해 각 weights 폴더에 `per_image_<split>.csv`(이미지별 7개 지표, scale_ratio, 유효 픽셀 수)를 저장하고 평균을 출력한다. 정답을 희소 형식으로 저장해 val 4,424장도 메모리 부담이 없다 |
| `test_eval_per_image.py` [PC] | 위 코드의 계산(정답 저장·복원, crop, scaling)을 numpy로 점검하는 테스트 |
| `README.md` [문서] | 채점 기준과 앞으로 할 분석 목록 |

### 6.4 `data/` — 데이터 목록

| 파일 | 설명 |
|---|---|
| `splits/train_files.txt` | 학습 목록 39,810줄 (Monodepth2 eigen_zhou). 한 줄 예: `2011_09_26/2011_09_26_drive_0022_sync 473 r` = 주행 폴더, 프레임 번호, 카메라(l 왼쪽 / r 오른쪽) |
| `splits/val_files.txt` | 검증 목록 4,424줄. epoch 고르기용 |
| `splits/test_files.txt` | 시험 목록 697줄 (Eigen split). 최종 점수용. 모두 왼쪽 카메라 |
| `splits/README.md` [문서] | 목록의 출처, 해시, 분석 결과(train/test 장면 분리, val의 반대 카메라 문제) |
| `kitti_archives_to_download.txt` | KITTI 다운로드 주소 66개(주행 61 + 보정값 5). `prepare_kitti.py`가 읽음 |
| `README.md` [문서] | 데이터 위치와 준비 방법 |

학습 코드가 실제로 읽는 목록은 `vendor/Lite-Mono/splits/`에 있는 같은 내용의 사본이다.

### 6.5 기타 폴더

| 파일 | 설명 |
|---|---|
| `environment/requirements-colab.txt` [Colab] | Colab 기본 설치(torch 등) 위에 추가할 패키지: tensorboardX(학습 기록), thop(연산량 측정), timm, scikit-image, opencv, matplotlib, 학습률 스케줄러(commit 고정) |
| `environment/README.md`, `server_info.example.md` [문서] | 환경 기록 방법, Colab 제약 |
| `configs/README.md`, `paths.env.example` [문서] | 서버 경로 설정 양식. **Colab에서는 쓰지 않는다**(경로는 노트북 1단계에서 정함) |
| `experiments/README.md`, `run_template.md` [문서] | 학습마다 복사해 채우는 실험 기록 양식 |
| `reports/proposal.md`, `intermediate.md`, `final.md` [문서] | 제출 보고서 초안 자리 |
| `reports/figures/README.md` [문서] | 보고서 그림 이름 규칙 |
| `research/lite_mono_review.md` [문서] | Lite-Mono와 관련 논문 정리, 개선 아이디어 근거 |
| `research/README.md`, `paper_note_template.md` [문서] | 논문 읽기 순서와 메모 양식 |

### 6.6 `vendor/` — Lite-Mono 공식 코드

| 파일 | 설명 |
|---|---|
| `vendor/README.md` [문서] | 공식 코드 출처(commit `4874b35`), 가중치 체크섬, 고친 곳과 이유 |
| `vendor/changes_from_upstream.patch` [문서] | 공식 코드 대비 바뀐 줄 전체(diff) |

`vendor/Lite-Mono/` 안의 파일. 고친 곳은 코드에 `# compat:`(최신 라이브러리 호환)과 `# added:`(Colab 운영 기능) 주석으로 표시되어 있다.

**학습·채점의 시작점**

| 파일 | 설명 |
|---|---|
| `train.py` [Colab] | 학습 시작 파일. 옵션을 읽고 `Trainer`를 만들어 `train()`을 부른다. 12줄 |
| `options.py` [Colab] | 모든 명령줄 옵션과 기본값(batch, epoch, lr, 해상도 등). `--seed` 추가 |
| `trainer.py` [Colab] | **학습의 본체.** 네트워크·옵티마이저·데이터 로더를 만들고, 매 스텝 "깊이·이동 예측 → 사진 다시 그리기 → 손실 → 수정"(4.2절)을 반복한다. 로그 출력, TensorBoard 기록, epoch마다 저장도 여기서 한다. 이어하기·진행률·저장 정리 기능을 추가했다 |
| `evaluate_depth.py` [Colab] | 공식 채점 코드. test 697장을 예측하고 `splits/eigen/gt_depths.npz`와 비교해 7개 지표 평균을 출력 |
| `export_gt_depth.py` [PC] | LiDAR로 공식 정답 `gt_depths.npz`를 만든다(Monodepth2에서 가져옴). PC에서 실행해 `eigen_gt_depths.npz`를 만들었다 |
| `test_simple.py` | 사진 한 장의 깊이를 예측해 이미지로 저장하는 데모. 노트북에서는 안 씀. 결과 그림을 만들 때 쓸 수 있다 |

**네트워크 `networks/`**

| 파일 | 설명 |
|---|---|
| `depth_encoder.py` [Colab] | **Lite-Mono 인코더**(`LiteMono` 클래스). 3단계 구조, 채널 48 → 80 → 128, 블록 4/4/10개. CDC 블록과 LGFI 블록이 여기 있다 |
| `depth_decoder.py` [Colab] | **깊이 디코더**. 특징을 3번 키우며 3개 해상도로 disparity(깊이의 역수, 0~1)를 출력. 개선 대상 |
| `resnet_encoder.py` [Colab] | PoseNet의 인코더(ResNet18). 사진 2장을 겹쳐 6채널로 받는다 |
| `pose_decoder.py` [Colab] | PoseNet의 출력부. 회전 3 + 이동 3 = 숫자 6개 |
| `__init__.py` | 위 클래스들을 `networks.XXX`로 부를 수 있게 함 |

**데이터 `datasets/`**

| 파일 | 설명 |
|---|---|
| `mono_dataset.py` [Colab] | 데이터 로더의 공통 부분. 목록 한 줄을 받아 t-1, t, t+1 사진을 읽고, 640×192로 줄이고, 색 흔들기·좌우 뒤집기 증강을 하고, 카메라 보정 행렬을 붙인다 |
| `kitti_dataset.py` [Colab] | KITTI 전용 부분. 파일 경로 규칙(`image_02` = 왼쪽, `image_03` = 오른쪽), 카메라 내부 행렬. LiDAR 파일이 있으면 정답도 읽지만, 학습 폴더에는 일부러 LiDAR를 넣지 않았다 |
| `__init__.py` | 데이터셋 클래스 내보내기. (이 폴더가 `.gitignore` 때문에 GitHub에서 빠졌다가 2026-11-09에 고쳤다) |

**도우미**

| 파일 | 설명 |
|---|---|
| `layers.py` [Colab] | 계산 부품: disparity → 깊이 변환, 회전·이동 행렬, 픽셀을 3D로 올리고(`BackprojectDepth`) 다른 카메라로 투영(`Project3D`) = "사진 다시 그리기"의 핵심, SSIM, 부드러움 손실, 학습 중 깊이 지표 |
| `kitti_utils.py` [PC·Colab] | LiDAR 점을 카메라 이미지 위 깊이 지도로 바꾸는 함수. 정답 만들기에 씀 |
| `utils.py` [Colab] | 파일 줄 읽기, 시간 표시 같은 작은 함수 |

**학습 목록 `splits/`**

| 파일 | 설명 |
|---|---|
| `eigen_zhou/train_files.txt`, `val_files.txt` [Colab] | 학습이 읽는 목록. `data/splits/`와 같은 내용 |
| `eigen/test_files.txt` [Colab] | 채점이 읽는 목록 |
| `eigen/gt_depths.npz` | Git에 없음. 4단계가 Drive에서 복사해 넣는다 |

**쓰지 않는 것**

| 파일 | 설명 |
|---|---|
| `lite-mono-pretrain-code/` | 저자가 ImageNet으로 인코더를 미리 학습할 때 쓴 코드. 우리는 결과 가중치(`lite-mono-pretrain.pth`)만 받아 쓰므로 실행하지 않는다 |
| `img/` | 공식 README의 그림 |
| `README.md`, `license`, `LICENSE_monodepth2`, `.gitignore` | 공식 저장소의 설명과 라이선스(Lite-Mono MIT, Monodepth2 비상업) |

### 6.7 Git에 없는 PC 파일

| 위치 | 설명 |
|---|---|
| `data/processed/` | Drive에 올린 4개 파일 원본 (7.2GB) |
| `data/raw/kitti_data/` | tar로 묶기 전의 사진 폴더 (6.4GB). tar와 내용이 같으므로 Drive 업로드를 확인했으면 지워도 된다 |
| `data/raw/kitti_velo/` | val/test LiDAR 원본 (8.9GB). 정답 npz를 다시 만들 때만 필요하다 |
| `pretrained/` | 공식 가중치의 PC 사본 |

---

## 7. Drive `lite_mono/` 폴더 (학습 후 모습)

```
lite_mono/
├── data/
│   ├── kitti_jpg.tar            사진 45,259장 + 보정값 (4단계가 풂)
│   ├── eigen_gt_depths.npz      test 정답, 공식 형식 (5·7단계 공식 채점)
│   ├── test_gt.npz              test 정답, 이미지별 형식 (5·7단계)
│   └── val_gt.npz               val 정답 (7단계 epoch 고르기)
├── pretrained/
│   ├── lite-mono-pretrain.pth   ImageNet 인코더 (학습 출발점)
│   └── lite-mono_640x192/       저자 KITTI 모델 (5단계 점검)
├── env/session_*.txt            세션별 GPU·패키지 기록
└── outputs/<RUN_ID>/
    ├── models/
    │   ├── opt.json             학습에 쓴 옵션 전체
    │   ├── weights_0/ … weights_29/   epoch별 모델 (+ 7단계 후 per_image_val_files.csv)
    ├── train/, val/             TensorBoard 기록
    ├── train_log.txt            화면 로그 전체
    ├── gpu_log.csv              1분마다 GPU 사용량
    ├── sessions.txt             세션 시작·끝 시각 (몇 번 끊겼는지)
    └── val_summary.txt          7-1 결과 요약
```

---

## 8. 자주 나는 문제

| 증상 | 원인과 해결 |
|---|---|
| 1단계 `ValueError: mount failed` | 로그인 팝업에서 권한을 일부만 허용함. 런타임 → 런타임 연결 해제 및 삭제 → 1단계 다시 실행 → 같은 계정, 체크박스 전부 선택 |
| 1단계 `... 에 없는 파일: [...]` | Drive 폴더 위치가 다름. `lite_mono`가 **내 드라이브 바로 아래**에 있어야 한다(`Colab Notebooks` 안 X). 이름에 번호가 붙은 중복 파일(`kitti_jpg (1).tar`)만 있고 원래 이름이 없는 경우도 같은 오류가 난다 |
| 1단계 `코드 받기 실패` | 위에 나온 git 오류를 확인. GitHub 저장소와 `Colab` 브랜치가 있는지 확인 |
| `cd: /content/Lite-Mono-Depth-Estimation: No such file or directory` | 1단계를 실행하지 않았거나 런타임이 초기화됨. 1단계부터 다시 |
| `module 'datasets' has no attribute 'KITTIRAWDataset'` | 코드 폴더 `vendor/Lite-Mono/datasets/`가 없음(2026-11-09 수정됨). 1단계를 다시 실행하면 최신 코드로 업데이트된다 |
| `NameError: name 'RUN_ID' is not defined` 등 | 런타임이 초기화되어 변수가 사라짐. 1단계부터 다시 |
| `CUDA out of memory` | 논문의 batch 12는 12GB GPU에서 돌았으므로 T4(16GB)에는 들어가야 함. 런타임을 다시 시작한 뒤 1 → 2 → 4 → 6. batch를 줄이면 논문 조건과 달라지므로 마지막 수단 |
| 학습 중 세션 끊김 | 정상적인 상황. 1 → 2 → 4 → 6을 다시 실행하면 이어서 한다 |
| 5단계 점수가 0.107과 다름 | 학습하지 말 것. 4단계 점검 출력, `eigen_gt_depths.npz` 복사 여부를 확인하고 출력 전체를 공유 |

코드를 고친 뒤에는 PC에서 GitHub `Colab` 브랜치에 push하고, Colab 1단계를 다시 실행하면 반영된다. 이 PC에서는 SSH로 push한다(`git push git@github.com:Sangbum-K/Lite-Mono-Depth-Estimation.git Colab`).
