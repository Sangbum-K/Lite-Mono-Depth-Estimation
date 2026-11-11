# Lite-Mono 공부 정리

정리일: 2026-11-11
대상 논문: **Lite-Mono: A Lightweight CNN and Transformer Architecture for Self-Supervised Monocular Depth Estimation** (CVPR 2023)

---

## 목차

1. 논문의 주제와 전체 흐름
2. 왜 Lite-Mono를 골랐나 (선행연구 비교)
3. 스테레오 방식과 무엇이 다른가
4. 연구 흐름 속 DepthNet·PoseNet 변화
5. ResNet으로 어떻게 깊이를 예측하나 (백본 개념)
6. 처음엔 깊이를 어떻게 아나 (학습 원리)
7. 절대 깊이 없이 어떻게 학습하나
8. 깊이 "차이"가 아니라 "비율"
9. Encoder와 Decoder의 역할
10. 테슬라 카메라 거리 추정과의 비교
11. Lite-Mono 구조 상세 (overview 그림 + Concat)

---

## 1. 논문의 주제와 전체 흐름

**Q. 이 논문은 무엇을 하나? 데이터셋, 학습, 백본까지 처음부터.**

### 문제 정의

- **단안 깊이 추정**: RGB 사진 **한 장**을 넣으면 픽셀마다 카메라에서의 거리를 출력한다.
- **자기지도 학습**: 학습할 때 정답 깊이(LiDAR)를 **쓰지 않는다**. 연속된 영상 프레임만 쓴다.
- **경량**: 3.1M 파라미터. Monodepth2-ResNet18(14.3M)보다 작은데 더 정확하다 (AbsRel 0.115 → 0.107).
- 핵심 기여는 **encoder 구조**(CNN + Transformer 혼합)다. 학습 방식과 손실은 Monodepth2에서 가져왔다.

### 데이터셋: KITTI

| 항목 | 내용 |
|---|---|
| 원본 | KITTI raw (카를스루에 주행 영상, 카메라 + LiDAR) |
| 학습/검증 | Eigen-Zhou split: train 39,810 / val 4,424 |
| 테스트 | Eigen split 697장, 정답은 LiDAR를 투영한 희소 깊이 |
| 입력 크기 | 640×192 |
| 학습 단위 | 연속 3프레임 (t−1, t, t+1) + 카메라 내부 파라미터 K |
| 증강 | 50% 좌우 반전, 50% color jitter. 3프레임에 **같은** 증강 적용 |

- Zhou split은 차가 멈춰 있는 정지 장면을 뺀 목록이다. 카메라가 움직여야 학습 신호가 생긴다.
- 논문 본문은 학습 수를 39,180으로 적었지만 공개 목록은 39,810이다 (오타로 추정).
- 내 로컬 실험은 JPG가 아니라 원본 PNG를 썼다 (`CHANGES.md` 참고).

### 학습 원리 (자기지도)

```
프레임 t ──► DepthNet ──► 깊이 D_t
프레임 t, t±1 ──► PoseNet ──► 카메라 이동 T
        │
        ▼
D_t + T + K로 "t의 각 픽셀이 t±1의 어디에 보일지" 계산
        │
        ▼
t±1에서 그 위치의 색을 가져와 t를 재합성 → Î_t
        │
        ▼
손실 = 실제 I_t 와 재합성 Î_t 의 차이 (photometric loss)
```

- 깊이와 카메라 이동이 **둘 다 맞아야** 재합성이 정확해진다.
- PoseNet은 **학습할 때만** 쓰고 추론 때는 DepthNet만 쓴다.
- 단안이라 **scale을 모른다** → 평가 때 median scaling으로 맞춘다.

### 모델 구조 요약

- **Depth Encoder**: CDC(연속 dilated conv, 지역 특징) + LGFI(XCA attention, 전역 특징). ImageNet 사전학습 가중치로 시작.
- **Depth Decoder**: 업샘플 → skip concat → conv 반복. 3개 스케일에서 disparity 출력 (sigmoid).
- **PoseNet**: Monodepth2와 같은 ResNet18. 두 프레임을 붙여 넣고 6-DoF(회전 3 + 이동 3) 출력.

### 손실 함수

| 구성 | 설명 |
|---|---|
| Photometric loss | `0.85 × SSIM 항 + 0.15 × L1` |
| Minimum reprojection | t−1, t+1 재합성 중 오차가 **작은 쪽만** 사용 → 가림(occlusion) 대응 |
| Auto-masking | 원본 인접 프레임이 더 잘 맞는 픽셀은 제외 → 같이 움직이는 차, 정지 구간 대응 |
| Edge-aware smoothness | 깊이를 매끄럽게, 단 이미지 경계에서는 약하게. 가중치 1e-3, 스케일마다 /2^s |
| Multi-scale | 3개 스케일 출력을 원본 크기로 키운 뒤 손실 계산 |

모두 Monodepth2에서 온 것이다.

### 학습 설정

| 항목 | 값 |
|---|---|
| batch / epoch | 12 / 30 |
| Optimizer | AdamW, weight decay 1e-2 |
| LR | Depth 1e-4 → 5e-6, Pose 1e-4 → 1e-5, cosine (31 epoch 주기) |
| Drop path | 0.2 |
| 로컬 소요 시간 | RTX 5070에서 약 9시간 |

### 평가

1. Eigen test 697장 예측
2. Garg crop (LiDAR가 없는 영역 제외)
3. 80m cap
4. **Median scaling**: 이미지마다 `median(GT) / median(예측)`을 곱함
5. 지표: AbsRel(주 지표), SqRel, RMSE, RMSE log (작을수록 좋음) / δ₁, δ₂, δ₃ (클수록 좋음)

| | AbsRel | SqRel | RMSE | δ₁ |
|---|---|---|---|---|
| 논문 | 0.107 | 0.765 | 4.561 | 0.886 |
| 내 재현 (`weights_29`) | 0.107 | 0.821 | 4.608 | 0.887 |

---

## 2. 왜 Lite-Mono를 골랐나

**Q. 다른 선행 연구들과 비교해서 왜 이 논문인가?**

### 같은 조건의 성능 비교 (640×192, KITTI Eigen, 단일 프레임 추론)

| 방법 | 파라미터 | AbsRel ↓ | RMSE ↓ | δ₁ ↑ | 의미 |
|---|---:|---:|---:|---:|---|
| Monodepth2-Res18 | 14.3M | 0.115 | 4.863 | 0.877 | 학습법의 출발점 |
| **Lite-Mono** | **3.1M** | **0.107** | **4.561** | **0.886** | 1/5 크기로 더 정확 |
| MonoViT-tiny | 10.3M | 0.102 | 4.459 | 0.895 | 더 정확하지만 3배 이상 큼 |
| Mono-ViFI + Lite-Mono | 3.1M | 0.103 | 4.349 | 0.891 | 같은 모델을 학습법만 바꿔 개선 |
| NimbleD + Lite-Mono | 3.1M | 0.096 | 4.304 | 0.903 | 큰 교사 모델 + 외부 영상 사용 |
| RTS-Mono-XS (preprint) | 2.4M | 0.101 | 4.357 | 0.897 | 최근 경량 연구, 게재 미확정 |

출처: Lite-Mono Table 2, Mono-ViFI ECCV 2024 Table 1, NimbleD 저자 결과표, RTS-Mono preprint Table II (`research/lite_mono_review.md`)

- Lite-Mono는 **정확도 1위(SOTA)가 아니다**. 강점은 **크기 대비 성능**이다.
- 같은 3.1M 모델에서 후속 연구가 0.103, 0.096까지 낮췄다 → **개선 여지가 문헌으로 확인된다**.

### 고른 이유 (노트에 기록된 것)

1. **과제 요건 충족**: CVPR 2023, 논문의 benchmark(KITTI)를 재현 가능. 과제는 SOTA가 아니라 "직접 재현한 baseline보다 개선"을 요구한다.
2. **12GB GPU로 여러 번 학습 가능**: 1회 약 9시간 → baseline, 개선안, 대조군을 같은 조건으로 반복할 수 있다.
3. **코드·가중치 공개, 재현 확인**: 실제로 0.107 재현.
4. **지금도 쓰이는 baseline**: 2024~2026 후속 연구의 출발점.
5. **수정 지점이 명확**: decoder가 단순해서 한 곳만 바꾸고 원인을 분리하기 좋다.


### 연구 방향과의 연결

- 학습법 개선 → Mono-ViFI가 했음 (비용 큼)
- 증류 → NimbleD가 했음 (외부 정보 필요)
- decoder 특징 결합 → HR-Depth가 큰 모델에서 효과. **경량 모델 decoder의 한 지점만** 바꾸는 실험은 추가 데이터 없이 가능

**연구 질문**: encoder와 학습법은 그대로 두고, decoder 특징 결합 한 곳만 바꿔서 경계 주변 깊이 번짐을 줄이고 AbsRel도 낮출 수 있는가?
→ "용량만 늘려서 좋아진 것" 반론을 막기 위해 **단순 conv 추가 대조군**도 함께 돌린다.

---

## 3. 스테레오 방식과 무엇이 다른가

**Q. 원래는 다른 카메라 두 장에서 correspondence를 찾아 깊이를 쟀는데, 이건 카메라 하나의 프레임 간 이동으로 깊이를 잰다는 거지?**

학습할 때는 맞다. 단, 추론할 때는 다르다.

| | 스테레오 | Lite-Mono (단안 영상) |
|---|---|---|
| 두 시점 | 같은 순간, 카메라 2대 | 다른 순간, 카메라 1대 |
| 카메라 사이 이동 | 고정·알려짐 (KITTI 약 54cm) | 모름 → PoseNet이 추정 |
| 단위 | 미터 | scale 모름 → median scaling 필요 |
| 가정 | 같은 순간이라 장면 동일 | 장면이 정지해 있다고 가정 → 움직이는 차에서 깨짐 |

- 기본 설정은 `frame_ids = [0, -1, 1]`이고 스테레오 프레임은 안 쓴다 (`--use_stereo`로 쓸 수는 있음).
- **Correspondence를 탐색하지 않는다**: 깊이를 먼저 예측 → 대응 위치를 계산 → 재합성 오차로 맞았는지 검사.
- **추론은 사진 한 장**: 프레임 간 움직임은 **학습 신호**로만 쓰이고, 테스트 때는 물체 크기, 도로 위 위치, 가림, 원근 같은 **단안 단서**로 깊이를 추론한다.
- 약점: 같은 속도로 앞서 달리는 차는 움직임이 없어 보여 "무한히 멀다"고 학습되기 쉽다.

---

## 4. 연구 흐름 속 DepthNet·PoseNet 변화

**Q. PoseNet과 DepthNet도 연구 흐름 표에 넣어서 정리.**

| 단계 | 대표 연구 | 학습 신호 | DepthNet | PoseNet | 한계 |
|---|---|---|---|---|---|
| ① 지도학습 | Eigen et al. (2014) | LiDAR 정답 | coarse + fine 두 단계 CNN | 없음 | 정답 수집 비용 |
| ② 자기지도 (스테레오) | Garg (2016) | 좌우 재합성 | encoder-decoder CNN | 없음 (기준선 고정) | 스테레오 필요 |
| | Monodepth (2017) | 좌우 재합성 + 일관성 | VGG / ResNet50 encoder-decoder | 없음 | 〃 |
| ③ 자기지도 (단안) | SfMLearner (2017) | 인접 프레임 재합성 | DispNet형 encoder-decoder | **처음 등장**: 작은 전용 CNN | 가림·움직이는 물체 |
| ④ 학습법 정리 | **Monodepth2** (2019) | min reproj, auto-mask, multi-scale | ResNet18 + U-Net decoder, 14.3M | **ResNet18** (6ch 입력) + PoseDecoder | 이후 표준 |
| ⑤ 큰 모델 | PackNet-SfM (2020) | Monodepth2 계열 | 3D conv 대형 망, 약 128M | SfMLearner형 ⚠️ | 너무 무거움 |
| | HR-Depth (2021) | Monodepth2 | ResNet18 유지, **decoder 재설계** | Monodepth2 | 크기 비슷 |
| | MonoViT (2022) | Monodepth2 | CNN + Transformer (MPViT) | Monodepth2 | 정확하지만 큼 |
| ⑥ **경량화** | **Lite-Mono** (2023) | Monodepth2 | **CDC + LGFI, 3.1M** | Monodepth2 (ResNet18) | 정확도 1위 아님 |
| ⑦ 후속 | Mono-ViFI (2024) | VFI 중간 프레임 + 일관성 | Lite-Mono 그대로 | VFI 네트워크 추가 ⚠️ | 학습 비용 증가 |
| | NimbleD (2024 WS) | 자기지도 + 교사 증류 | Lite-Mono 그대로 | ⚠️ | 외부 데이터 필요 |
| 이번 과제 | Lite-Mono + decoder 수정 | Monodepth2 | encoder 유지, **decoder 한 곳 수정** | Monodepth2 | 검증 필요 |

- **PoseNet은 Monodepth2의 ResNet18 형태로 굳은 뒤 거의 안 바뀌었다.** 추론 크기에 안 들어가니 경쟁은 DepthNet에서 일어났다.
- DepthNet은 encoder 키우기(PackNet, MonoViT) / decoder 다듬기(HR-Depth) / encoder 경량화(Lite-Mono)로 갈렸다.
- 후속 연구는 DepthNet은 그대로 두고 **학습 신호**를 바꿨다.
- Lite-Mono, Monodepth2 행 외의 세부 구조는 ⚠️ 기억 기반이므로 원문 Method 절로 확인한다.

---

## 5. ResNet으로 어떻게 깊이를 예측하나

**Q. ResNet은 분류에서만 배웠는데, ResNet 아키텍처를 썼다는 건가?**

ResNet 전체가 아니라 **앞부분(특징 추출기)만 가져와 encoder로 쓴다**. 깊이 지도는 뒤에 붙인 **decoder**가 만든다.

### 백본(backbone) 개념

분류 ResNet은 두 부분으로 나뉜다.

```
이미지 → conv1 → layer1~4  →  avgpool → fc → 클래스 1000개
         └ ① 특징 추출 ┘       └─ ② 분류 head ─┘
```

- ①이 진짜 어려운 일 ("보는 눈"): 경계 → 질감 → 부품 → 물체
- ②는 단순한 "답하는 입"
- ①만 떼어 다른 작업의 기반으로 쓰는 것이 **백본**이다.

### 전이학습

1. ImageNet 가중치를 가져옴 (pretrained)
2. 분류 head 버림
3. 작업에 맞는 새 head 붙임
4. 내 데이터로 다시 학습 (fine-tuning)

| 작업 | 백본 | 새 head | 출력 |
|---|---|---|---|
| 분류 | ResNet | avgpool + fc | 클래스 |
| 검출 | ResNet | 박스 head | 박스 + 클래스 |
| 분할 | ResNet | decoder | 픽셀마다 클래스 |
| **깊이** | ResNet | **decoder** | **픽셀마다 깊이** |
| PoseNet | ResNet | conv + 평균 | 숫자 6개 |

### ResNet18 중간 출력 (640×192 입력)

| 출력 | 위치 | 크기 | 채널 | 정보 |
|---|---|---|---|---|
| f0 | conv1 | 1/2 (320×96) | 64 | 경계, 질감 |
| f1 | layer1 | 1/4 | 64 | |
| f2 | layer2 | 1/8 | 128 | |
| f3 | layer3 | 1/16 | 256 | |
| f4 | layer4 | 1/32 (20×6) | 512 | 의미 정보 |

코드 (`networks/resnet_encoder.py`)도 layer4까지만 호출하고 avgpool·fc는 부르지 않는다.

### 왜 decoder가 필요한가

- 깊이는 **dense prediction**: 입력과 같은 크기의 지도를 출력해야 한다.
- 백본을 지나면 1/32로 줄어든다 → **업샘플 + conv**로 다시 키우는 decoder가 필요.

### Skip connection

- 줄였다 키우기만 하면 흐릿해진다 (세부 위치 정보 손실).
- encoder의 **같은 크기 중간 특징을 decoder에 붙인다** (concat) → U-Net 구조.
- 깊은 특징 = **무엇** / 얕은 특징 = **어디** → 합치면 의미도 맞고 경계도 날카롭다.
- ResNet 안의 residual(`x + F(x)`, 블록 안 더하기)과 다르다. skip은 encoder → decoder로 건너뛰어 **붙이기**.
- **이 skip을 어떻게 합치느냐가 이번 과제의 개선 지점이다.**

### PoseNet에서의 ResNet

1. 첫 conv를 3ch → **6ch**로 바꿈 (두 프레임). ImageNet 가중치를 두 번 복사하고 2로 나눠 초기화.
2. 마지막 특징 f4만 사용 → PoseDecoder (`networks/pose_decoder.py`)
3. **공간 평균** (`out.mean(3).mean(2)`) → 숫자 6개
4. 회전 3 + 이동 3, 학습 안정을 위해 **×0.01**

### 직접 확인 코드

```python
import torch, torchvision

resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")
x = torch.randn(1, 3, 192, 640)

print(resnet(x).shape)            # [1, 1000]  분류로 사용

f0 = resnet.relu(resnet.bn1(resnet.conv1(x)))
f1 = resnet.layer1(resnet.maxpool(f0))
f2 = resnet.layer2(f1)
f3 = resnet.layer3(f2)
f4 = resnet.layer4(f3)
for f in [f0, f1, f2, f3, f4]:
    print(f.shape)
# [1,64,96,320] [1,64,48,160] [1,128,24,80] [1,256,12,40] [1,512,6,20]
```

**요약**: ResNet = "보는 눈"(백본) + "답하는 입"(head). Lite-Mono는 **눈을 ResNet보다 훨씬 작은 CDC + LGFI로 바꾼** 모델이다.

---

## 6. 처음엔 깊이를 어떻게 아나

**Q. 재합성 오차를 줄이라고 해도 처음 픽셀마다의 깊이 값은 어떻게 아나?**

**처음엔 모른다.** 아무 값에서 시작해 틀린 정도를 보고 고쳐 나간다. 분류 모델이 처음에 아무 클래스나 찍는 것과 같다.

### 실제 초기값

- decoder 마지막 conv: 작은 랜덤 가중치(std 0.02), bias 0 → 출력 ≈ 0 → sigmoid ≈ **0.5**
- `disp_to_depth`: disparity ≈ 0.01 + 9.99 × 0.5 ≈ 5 → 깊이 ≈ **0.2 (단위 없음)**
- → **모든 픽셀이 거의 같은 깊이** (평평한 벽)
- PoseNet은 ×0.01 → "거의 안 움직였다"

### 고칠 방향을 아는 원리: 재합성이 미분 가능하다

```
깊이 D(p) → 3D 점 → 카메라 이동 적용 → 인접 프레임에 투영 (p')
→ p' 위치 색을 bilinear로 읽음 → Î(p) vs I(p)
```

모든 단계가 미분 가능해서 "D(p)를 바꾸면 손실이 어떻게 변하나"를 계산할 수 있다.

```
∂Loss/∂D(p) = (Î(p) − I(p)) × ∇I_인접(p') × ∂p'/∂D(p)
               오차              읽은 위치 주변 색 변화   깊이 변화 → 위치 변화
```

1차원 예시: 깊이를 너무 멀게 예측 → 시차가 작음 → 회색을 읽음 → 주변 색 변화를 보니 "조금 더 이동하면 빨강" → 깊이를 **가깝게** 하는 방향으로 기울기.

- 이 미분 가능한 샘플링은 Spatial Transformer Network(2015) 아이디어. 코드에서는 `F.grid_sample` (`trainer.py`의 `generate_images_pred`).

### 깊이와 이동 둘 다 모르는데? (닭과 달걀)

- 하나의 손실에서 **둘 다 동시에** 기울기를 받는다.
- 이동은 이미지 쌍마다 숫자 6개, 픽셀은 12만 개 → 잘못된 이동으로는 대부분 픽셀이 동시에 맞을 수 없다 → 이동이 먼저 자리 잡고 깊이가 따라온다.

### 기울기만으로 부족한 곳과 해결 장치

| 문제 | 원인 | 해결 |
|---|---|---|
| 정답이 너무 멀리 있음 | 근처 픽셀만 봄 | Multi-scale (작은 해상도는 넓은 범위를 봄) |
| 질감 없는 영역 (벽, 하늘) | 색이 같아 기울기 0 | Smoothness loss |
| 가려진 픽셀 | 인접 프레임에 안 보임 | Min reprojection |
| 같이 움직이는 차 | 움직임 없어 보임 | Auto-masking |

→ 손실 구성들은 전부 "재합성 기울기만으로 부족한 곳"을 막는 장치다.

### 이미지 하나가 아니라 "규칙"을 배운다

- 네트워크 가중치는 하나, 4만 장 × 30 epoch.
- 여러 이미지에서 반복되는 패턴 → "이런 모양이면 이 정도 거리"라는 **일반 규칙** 학습.
- 그래서 테스트 때 인접 프레임 없이 한 장만으로 깊이를 낸다.

---

## 7. 절대 깊이 없이 어떻게 학습하나

**Q. 절대 깊이는 LiDAR가 있어야만 아는데 어떻게 비지도 학습이 되나?**

맞다. **이 방법은 절대 깊이를 배우지 못한다.** 배우지 못해도 되도록 문제와 평가가 설계돼 있다.

### 배우는 것 / 못 배우는 것

"모든 깊이 ×k, 카메라 이동 ×k"를 해도 재합성 결과가 같다. 그래서:

- ✅ 아는 것: 장면의 **모양** ("A가 B보다 2배 멀다")
- ❌ 모르는 것: **전체에 곱해진 숫자 하나 (scale)**
- 비유: **축척이 빠진 지도**

### LiDAR는 평가에서만

| 단계 | LiDAR |
|---|---|
| 학습 | ❌ 안 씀 |
| 평가 | ✅ 정답 비교 + scale 맞추기 |

```
ratio = median(LiDAR 깊이) / median(예측 깊이)
예측 × ratio → 지표 계산
```

- Eigen 점수 = "축척은 맞춰 줄 테니 **모양**이 얼마나 정확한가"
- 그래서 자기지도 단안 방법은 스테레오·GT 방법과 **표를 나눠** 비교한다.
- median scaling 평가는 실제 차량에서 미터 단위 거리를 얻는 성능과 **다르다** → 보고서에 한계로 밝힌다.

### 용어

- 비지도(unsupervised): 정답 신호 없음 (군집화 등)
- **자기지도(self-supervised)**: 사람이 만든 정답은 없지만 데이터에서 정답 신호를 만든다 (다음 프레임 이미지가 정답 역할). 논문 제목도 Self-Supervised.

### scale은 실제로 거의 일정하다

- 내 재현: median scaling ratio **32.49 ± 0.087** (± 값은 ratio/중앙값의 표준편차 → 이미지마다 약 9%만 흔들림)
- KITTI는 같은 차, 같은 카메라 높이, 비슷한 속도 → 네트워크가 일관된 내부 단위(1 ≈ 32m)를 스스로 정한다.
- 다른 차량·카메라에서는 이 상수가 안 맞는다.

### 절대 깊이를 얻는 방법

| 방법 | 알려진 실제 길이 | 예 |
|---|---|---|
| 스테레오 학습 | 좌우 카메라 거리 | Monodepth, Monodepth2 S/MS |
| 속도 정보 | 프레임 간 이동 거리 | PackNet-SfM velocity supervision |
| 카메라 높이 | 지면 위 높이 | 도로 평면 기반 scale 보정 |
| GT fine-tuning | LiDAR 일부 | metric depth 모델 |

이번 과제는 원 논문과 같이 **단안 + median scaling**을 유지한다 (공정 비교).

---

## 8. 깊이 "차이"가 아니라 "비율"

**Q. 그럼 깊이 차이를 측정하는 모델인 거네?**

정확히는 **깊이 비율(상대 깊이)을 추정하는 모델**이다.

예: 예측 A = 1, B = 2

| scale | A | B | 차이 | 비율 |
|---|---|---|---|---|
| ×10 | 10m | 20m | 10m | 2 |
| ×30 | 30m | 60m | 30m | 2 |

- **차이**는 scale에 따라 달라진다 → 모름
- **비율**은 항상 2 → 앎
- 용어: **relative depth**, **up-to-scale depth**
- "측정(measurement)"은 LiDAR처럼 실제로 재는 것, 이 모델은 **추정(estimation)**.

---

## 9. Encoder와 Decoder의 역할

**Q. 인코더와 디코더가 뭐고 여기서 하는 역할은?**

- **Encoder**: 이미지를 **줄이면서** 무엇이 있는지 특징으로 압축·이해
- **Decoder**: 특징을 **키우면서** 원하는 출력(픽셀별 깊이)으로 풀어냄
- 비유: encoder는 요약 메모를 쓰는 사람, decoder는 메모와 원본 세부를 참고해 깊이 지도를 그리는 사람

### Lite-Mono 실제 흐름 (640×192)

```
입력 3×192×640
 ── ENCODER (CDC + LGFI, 2.842M) ──
 stem          48×48×160
 Stage 1       48×48×160 → feature[0] ──────────────┐ skip
 Stage 2       80×24×80  → feature[1] ──────┐ skip   │
 Stage 3      128×12×40  → feature[2]        │        │
 ── DECODER (0.227M) ──                      │        │
 conv 128→64, 업샘플 24×80, + feature[1] ◄───┘        │
   conv → disp (scale 2) → 48×160                     │
 conv 64→40, 업샘플 48×160, + feature[0] ◄────────────┘
   conv → disp (scale 1) → 96×320
 conv 40→24, 업샘플 96×320 (skip 없음)
   conv → disp (scale 0) → 192×640 ← 최종 출력
```

(여기서 Stage 번호는 코드 기준. 그림 기준 번호는 11절 참고)

### Encoder 역할 (`networks/depth_encoder.py`)

1. 해상도 ↓, 채널 ↑ (192×640×3 → 12×40×128)
2. CDC: 지역 특징 (경계, 질감, 모양)
3. LGFI: 전역 특징 ("아래는 도로라 가깝다")
4. 중간 결과 3개를 decoder로 전달
- 논문 기여의 대부분, 모델 크기의 93%

### Decoder 역할 (`networks/depth_decoder.py`)

1. 해상도 복원 (2배씩 업샘플)
2. skip concat으로 세부 정보 회복
3. conv로 채널 1개 + sigmoid → disparity
4. 3개 스케일 출력 (학습 시 모두 사용, 추론 시 scale 0만)
- 업샘플 → concat → conv만 하는 단순 구조, 0.23M

### 과제에서의 의미

- Encoder: 논문 핵심 + ImageNet 가중치에 묶임 → **그대로 둠**
- Decoder: 단순, 경계 번짐이 생기는 곳, 특히 **skip을 그냥 concat하는 지점** → **개선 대상**
- PoseNet도 encoder(ResNet18) + decoder(PoseDecoder) 구조. decoder의 더 일반적 정의는 "특징을 원하는 출력 형태로 바꾸는 부분".

---

## 10. 테슬라 카메라 거리 추정과의 비교

**Q. 테슬라가 카메라로만 거리를 재는 것도 비슷한 원리인가?**

⚠️ 테슬라 내부 구현은 비공개. 아래는 AI Day(2021, 2022) 등 공개 발표·보도 기준이고 이후 바뀌었을 수 있다.

### 비슷한 점

- 신경망이 이미지에서 거리를 추론
- 백본 + head 구조 (AI Day: RegNet + BiFPN)
- 단안 단서 학습
- 연속 프레임 활용

### 다른 점

| | Lite-Mono | 테슬라 (공개 발표 기준) |
|---|---|---|
| 카메라 | 1대 | 8대, 시야 겹침 |
| 추론 입력 | 사진 1장 | 여러 카메라 × 여러 프레임 |
| 출력 | 픽셀별 상대 깊이 | BEV 3D 공간, occupancy, 물체 위치·속도 |
| 학습 신호 | 재합성 오차 (자기지도) | 주로 지도학습 (오프라인 자동 라벨링) |
| 데이터 | KITTI 약 4만 장 | 대규모 fleet 영상 |
| 절대 거리 | 모름 | 알 수 있음 |

### 테슬라가 절대 거리를 아는 이유

1. **차량 주행 정보**: 바퀴 속도 + IMU → 실제 이동 거리 (Lite-Mono는 PoseNet으로 추정하는 것을 센서로 측정)
2. **보정된 카메라 위치**: 높이, 카메라 간 거리. 겹치는 시야는 스테레오처럼 사용 가능
3. **오프라인 자동 라벨링**: 여러 차량 영상을 서버에서 3D 재구성 → 정답으로 지도학습
4. **검증용 LiDAR 차량**: 학습·검증용으로 사용한다는 보도

→ "카메라만 쓴다"는 **판매 차량 주행 시 센서**가 카메라뿐이라는 뜻이지, 학습까지 정답 없이 한다는 뜻이 아니다.

### 정리

- 원리의 뿌리는 같다.
- 테슬라는 scale 모호성(속도·보정), 한 장의 한계(다중 카메라·프레임), 자기지도의 부정확함(대규모 라벨)을 공학적으로 메웠다.
- 테슬라 방식의 정확도는 공개 벤치마크로 검증되지 않았다.
- 보고서 응용 동기로 쓸 수 있지만 "이번 결과가 자율주행 성능 향상을 입증한다"고 주장하지 않는다.

---

## 11. Lite-Mono 구조 상세 (overview 그림 + Concat)

**Q. overview 그림을 보면서 구조를 자세히. Concatenate가 뭔가?**

그림: `vendor/Lite-Mono/img/overview.png` (논문 Figure 2) — 노션에는 이 이미지를 따로 첨부.

### 그림과 코드의 Stage 번호

| 그림 | 코드 | 출력 크기 |
|---|---|---|
| Stage 1 (Conv stem) | `stem1` | H/2 × W/2 × 48 (96×320) |
| Stage 2 | `stem2` + `stages[0]` | H/4 × 48 (48×160) |
| Stage 3 | `downsample_layers[1]` + `stages[1]` | H/8 × 80 (24×80) |
| Stage 4 | `downsample_layers[2]` + `stages[2]` | H/16 × 128 (12×40) |

C₁~C₄ = 48, 48, 80, 128

### Concatenate란

**텐서를 채널 방향으로 이어 붙이는 것.** 값을 섞지 않고 옆에 쌓기만 한다.

```
A: [B, 48, 96, 320]
B: [B,  3, 96, 320]
torch.cat([A, B], dim=1) → [B, 51, 96, 320]
```

| | Concat | Add (residual) |
|---|---|---|
| 연산 | 채널 이어 붙이기 | 같은 위치 값 더하기 |
| 채널 수 | 늘어남 | 그대로 |
| 조건 | 가로·세로만 같으면 됨 | 모양이 완전히 같아야 함 |
| 정보 | 따로 보존, 다음 conv가 섞는 법을 학습 | 바로 합쳐짐 |

→ concat 뒤에는 거의 항상 **conv가 와서 채널을 줄인다** (그림: Concat → Downsample).

### 그림 속 연결 세 종류 (모두 concat)

1. **파란 점선 – Pooled Concatenation**: 입력 이미지를 AvgPool로 1/2, 1/4, 1/8로 줄여 각 단계에 RGB 3채널로 다시 붙임. 작은 모델에서 사라지는 원본 세부 정보를 다시 보여주기 위함.
2. **분홍 화살표 – Cross-Stage Connection**: 각 stage의 **Downsample 직후(CDC 전)** 특징을 다음 Concat으로 보냄. 검은 실선은 LGFI까지 통과한 후 특징. → "통과 전 + 통과 후"를 함께 전달.
3. **검은 점선 – Skip Connection**: encoder Stage 2, 3 출력을 decoder 같은 크기 단계에 붙임 (U-Net식).

### Concat 지점별 채널 계산

| 그림의 Concat | 붙이는 것 | 합 | Downsample 후 |
|---|---|---|---|
| Stage 1 → 2 | stem 48 + 이미지/2 3 | 51 | 48ch, H/4 |
| Stage 2 → 3 | 분홍 48 + 검정 48 + 이미지/4 3 | 99 | 80ch, H/8 |
| Stage 3 → 4 | 분홍 80 + 검정 80 + 이미지/8 3 | 163 | 128ch, H/16 |

코드의 `Conv(self.dims[i]*2+3, ...)`가 이 "×2 + 3"이다.

### Encoder 전체 흐름

```
I_t 3×192×640
 │ Stage 1 Conv stem: 3×3 conv(s2) → 3×3 → 3×3
 ▼ 48×96×320
 Concat(+이미지/2) → 51 → Downsample → 48×48×160 ─(분홍)
 │ Stage 2: CDC×3 (dilation 1,2,3) → LGFI → 48×48×160 ─(검정, skip)
 ▼
 Concat(분홍+검정+이미지/4) → 99 → Downsample → 80×24×80 ─(분홍)
 │ Stage 3: CDC×3 (1,2,3) → LGFI → 80×24×80 ─(검정, skip)
 ▼
 Concat(분홍+검정+이미지/8) → 163 → Downsample → 128×12×40
 │ Stage 4: CDC×9 (1,2,3,1,2,3,2,4,6) → LGFI
 ▼ 128×12×40 → decoder 시작
```

N₁, N₂, N₃ = 3, 3, 9 (코드 `self.depth = [4, 4, 10]`, 마지막 1개가 LGFI)

### CDC 블록 (주황, 코드 `DilatedConv`)

```
x ─┬─► 3×3 depthwise dilated conv → BatchNorm → 1×1 확장(6C) → GELU → 1×1 축소(C) → ×γ → DropPath ─┐
   └──────────────────────────────────── residual ─────────────────────────────────────────────── + → 출력
```

- **depthwise**: 채널마다 따로 conv → 파라미터 적음
- **dilated**: 커널 칸을 벌림. dilation 3이면 3×3 커널이 7×7 영역을 봄. 1, 2, 3 연속으로 쌓아 비용 그대로 시야 확대
- **1×1 확장·축소**: 채널끼리 섞기 (inverted bottleneck)
- **γ (layer scale)**: 1e-6으로 시작 → 학습 안정
- 정정: CDC 안의 정규화는 **BatchNorm**이다. LayerNorm(`self.norm`)은 정의만 되고 forward에서 안 쓰인다.

### LGFI 블록 (파랑)

```
x [B,C,H,W] → 펼치기 [B, H·W, C] → (+위치 인코딩, 그림 Stage 2만)
 ─┬─► LayerNorm → XCA → ×γ ─┐
  └──────────────────────── + → LayerNorm → 1×1 확장 → GELU → 1×1 축소 → ×γ → DropPath → + x
```

전반부 = 전역 attention, 후반부 = 채널 섞기.

**XCA가 싼 이유 (그림 Stage 2 기준)**

- 토큰 수: 48 × 160 = 7,680
- 일반 self-attention: 7,680 × 7,680 ≈ 5,900만 칸 (해상도 제곱으로 증가)
- XCA: **채널끼리** 계산. 48채널 / head 8 = head당 6 → **6 × 6 = 36칸** (해상도에 선형)
- 각 채널이 전체 픽셀의 요약이라 여전히 이미지 전체를 본다.

### Decoder

- **Upsampling Block (초록)**: 3×3 conv (채널 절반) → 2배 업샘플 → [skip concat] → 3×3 conv. 코드 `ConvBlock` = reflection pad + 3×3 conv + **ELU**
- **Prediction Head (노랑)**: conv (→ 1채널) → 2배 업샘플 → Sigmoid. 출력 H/4, H/2, H
- 그림과 코드 차이: 그림은 Prediction Head를 **1×1 conv**로 그렸지만 코드(`Conv3x3`)는 **3×3 conv**. 보고서는 코드 기준으로 쓴다.

### PoseNet과 학습 (그림 아래)

- PoseNet: I_t, I_s → ResNet-18 → P (6-DoF)
- Image reconstruction: 깊이 + P로 I_s에서 I_t 재합성
- Optimization: 재합성 오차로 두 네트워크 함께 학습

### 요약

Lite-Mono encoder = **CDC로 넓게 보고, LGFI로 전체를 보고, 줄어들 때마다 Concat으로 원본 이미지와 이전 특징을 다시 붙여 정보 손실을 막는 구조.** 과제의 개선 지점은 **decoder 쪽 skip Concat** (검은 점선이 초록 상자로 들어가는 곳).
