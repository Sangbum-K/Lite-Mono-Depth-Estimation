# Depth Pro 2 분석과 Lite-Mono 개선안 후보

작성: 2026-11-23. 작업 메모이며 보고서 원고가 아니다.

## 1. 출처

- 논문: P. Hu, M. Mihajlovic, L. Duong, L. Lipson, W. Dong, A. Delaunoy, S. R. Richter, V. Koltun. "Depth Pro 2: Sharp and Temporally Consistent Metric Monocular Depth." arXiv:2611.26603, 2026-11-19 (Apple). 학회 미기재.
- 코드: github.com/apple-aiml-research/ml-depth-pro-2, commit `aa389cb` (2026-11-19). 로컬 `third_party/ml-depth-pro-2/`에 받아 두었다(gitignore, 읽기 전용).
- 라이선스: 코드는 Apple 샘플 코드 라이선스, 가중치는 **연구 목적만 허용**(Apple ML Research Model license). 수업 과제는 연구 목적이라 사용할 수 있지만 출처를 반드시 표기한다.
- 저장소에는 **추론 코드만 있다.** 학습·평가 코드는 없다. 학습과 손실 관련 내용은 논문에서만 확인할 수 있다.
- 논문 확인 범위: arXiv HTML 앞부분(초록~Table 4 직전)은 원문으로 확인했다. 부록의 KITTI AbsRel 표와 속도 표는 요약본으로만 봐서 **미검증**이다.
- 선행 연구: Depth Pro (Bochkovskii et al., "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second", arXiv:2410.02073, 2024).

## 2. 우리 과제와 무엇이 다른가

| 항목 | Lite-Mono (우리 baseline) | Depth Pro 2 |
|---|---|---|
| 학습 신호 | **자기지도**: 인접 프레임 재투영 photometric loss, 깊이 GT 사용 안 함 | **지도학습**: 19개 데이터셋의 깊이 GT. 합성 데이터 비중이 큼 |
| 출력 | 상대 깊이 (평가 때 median scaling) | metric 깊이 (focal로 정규화한 inverse depth) |
| 크기 | 3.1M | 24M (S) ~ 872M (H+) |
| encoder | CNN + LGFI (CDC) | DINOv3 ViT, patch 16 |
| 입력 해상도 | 640×192 고정 | 원본 해상도 그대로 (경계 선명도가 처리 해상도와 비례한다고 주장) |
| 연산 자원 | GPU 1장 | B200 128장, 100K step (요약본, 미검증) |

같은 문제("한 장의 영상에서 깊이 추정")를 다루지만 **조건이 전혀 달라서 직접 경쟁 상대는 아니다.** 그대로 재구현할 대상이 아니라 아이디어를 가져올 참고 자료로 쓴다.

## 3. 코드에서 확인한 구조 (`src/depth_pro_2/network/`)

- `encoder.py` `MultiViewEncoder`: ViT의 4개 block(S 기준 2·5·8·11번) 출력을 꺼낸다. 이 출력은 모두 1/16 해상도의 token이다. 각 출력을 **transposed conv로 1/4·1/8, 그대로 1/16, stride-2 conv로 1/32**로 바꿔 4단계 pyramid를 만든다(DPT reassemble).
- `decoder.py` `DPTDecoder` / `FeatureFusionBlock2d`: 각 단계에서 `x + ResBlock(skip)`, 이어서 `ResBlock`, **`ConvTranspose2d(k=2, s=2)`로 2배 upsample**, 1×1 conv를 거친다. bilinear upsample은 쓰지 않는다. 결합 방식은 concat이 아니라 덧셈이다.
- `create_inverse_depth_head`: 1/2 해상도 특징을 conv → **ConvTranspose(×2)** → conv → ReLU 순으로 처리해 원본 해상도 inverse depth를 만든다. 마지막 upsample도 학습되는 연산이다.
- 영상(video) 처리: 뒤쪽 2/3 block에서 프레임 간 attention을 쓴다(S는 5·7·9·11번 block). 우리 과제(한 장 추론)에는 해당하지 않는다.

## 4. Lite-Mono의 해상도 흐름 (직접 확인, 640×192 입력)

- encoder 출력: 1/4 (48ch, 48×160), 1/8 (80ch, 24×80), 1/16 (128ch, 12×40)
- **stem1이 1/2 해상도 특징(48ch, 96×320)을 계산하지만 decoder로 넘기지 않고 버린다** (`networks/depth_encoder.py:345-415`).
- decoder (`networks/depth_decoder.py`): 매 단계 bilinear ×2 → concat skip → conv. 최종 disp는 **1/2 해상도에서 skip 없이** 만든 뒤 **bilinear ×2로 원본 해상도에 맞춘다.**
- 즉 깊이 경계를 결정하는 가장 세밀한 encoder 정보가 1/4 해상도다. 1/2와 원본 해상도로 올리는 과정은 학습되지 않는 bilinear 보간이다.
- 평가에서는 이렇게 나온 640×192 disparity를 KITTI 원본 크기(약 1242×375)로 한 번 더 bilinear resize한다.

## 5. 우리 실패 분석과 연결 (`outputs/20261109_baseline_unseeded/analysis/weights_29/`)

- 깊이 경계(GT 9×9 이웃 안에서 깊이 1.25배 이상 차이) 근처는 점의 11.5%인데 AbsRel 합의 28.1%를 차지한다.
- **배경 쪽 점은 평균 29% 가깝게 예측한다** (log −0.348, δ₁ 0.396). **앞 물체 쪽 점은 18% 멀게 예측한다** (log +0.162, δ₁ 0.665). 두 깊이가 경계를 넘어 섞이는 번짐(edge fattening)이다.
- 상위 30% 이미지 209장을 직접 보면, 주차된 차량 윗선과 옆선을 따라 파란 띠(배경을 너무 가깝게 예측)가 거의 모든 차량에서 보인다.
- 주의: LiDAR와 카메라 위치가 달라서 생기는 GT 오류도 경계 배경 쪽에 몰린다. 이 효과를 완전히 분리하지는 못했다(7절의 확인 실험 참고).
- (2026-11-27) 픽셀 단위 측정 (`analysis/edge_offset.py`): 깔끔한 LiDAR 경계 3,263개 중 예측 경계가 배경 쪽으로 4px 넘게 밀린 경우가 27.7%, 반대쪽은 7.1%다. 위치를 측정한 경계의 중앙값은 +3.5 GT px(네트워크 출력 기준 약 2px)다. 앞 물체를 아예 놓친 경우도 22.9%다. A안의 효과는 이 지표들(번짐 비율, 놓침 비율)로 직접 비교할 수 있다.

Depth Pro 2 논문의 관찰("깊이 불연속에서 평균을 내면 flying pixel이 생긴다", "경계 선명도는 처리 해상도와 비례한다", "bilinear 대신 transposed conv")이 우리가 관찰한 번짐과 같은 현상을 가리킨다.

## 6. 개선안 후보

| # | 아이디어 | 출처 | 목표 실패 | 추가 비용 | 범위 판단 |
|---|---|---|---|---|---|
| A | **decoder 고해상도 경로**: stem의 1/2 특징을 skip으로 연결하고, 마지막 두 upsample(1/2→1, 그리고 각 단계)을 bilinear에서 학습형(ConvTranspose 또는 PixelShuffle)으로 바꾼다 | Depth Pro / Depth Pro 2의 transposed-conv upsampling과 DPT fusion을 **변형해서 적용**. 고해상도 skip은 HR-Depth (Lyu et al., AAAI 2021)와 같은 계열 | 경계 번짐 (오차 비중 28%) | 파라미터 수십 K 수준(구현 후 실측), encoder는 그대로 | ✅ PLAN의 "decoder 특징 결합 한 부분 변경"과 일치 |
| B | A에 더해 skip을 concat이 아닌 `x + ResBlock(skip)` 형태의 residual fusion으로 결합 | Depth Pro 2 `FeatureFusionBlock2d` | 경계 번짐 | 작음 | ✅ ablation 후보 (A와 B 비교) |
| C | edge-aware smoothness가 경계를 넘어 평활화하는지 점검 (가중치 조정 또는 경계 영역 완화) | monodepth2 손실 분석 + Depth Pro 2의 "불연속 평균 금지" 관찰 | 경계 번짐 | 없음 | △ 손실 변경이라 구조 가설과 섞이지 않게 별도로 다룬다 |
| D | Depth Pro 2-S를 teacher로 쓰는 pseudo-label 증류 (scale-invariant / gradient matching loss) | Depth Pro 2의 L_ssi, L_mge | 경계, 원거리 | 오프라인 pseudo-label 생성 | ❌ PLAN 범위 밖(큰 teacher·외부 데이터). Depth Pro 2는 Waymo·VKITTI2 등으로 학습해 KITTI와 가까운 데이터를 이미 봤을 수 있어 비교의 공정성 문제도 있다. 선택하려면 사용자 결정이 필요 |

추천: **A를 주 가설로 하고 B를 ablation으로 둔다.** 이유는 세 가지다.
- 실패 분석(경계 번짐)과 원인(1/2·원본 해상도에서 bilinear만 쓰고 1/2 특징을 버림)이 코드 수준에서 직접 연결된다.
- 기존 연구(Depth Pro/2, HR-Depth)를 인용할 수 있다.
- 추가 비용이 작아 5 epoch pilot을 바로 돌릴 수 있다.

Lite-Mono를 가볍게 만든 원칙(3.1M)도 지킬 수 있다. A가 비용 대비 효과가 있는지는 파라미터, FLOPs, 지연시간을 함께 보고해서 보인다.

## 7. Depth Pro 2 저장소를 분석에 쓰는 방법 (선택)

- **GT 오류 확인용 비교**: Depth Pro 2-S(24M, 97MB)로 Eigen test 697장을 추론하고, 같은 median scaling으로 경계 배경 쪽 오차를 계산한다.
  - 경계가 선명한 모델도 같은 위치에서 큰 오차를 보이면, 그 부분은 LiDAR GT 오류일 가능성이 높다.
  - 학습에는 쓰지 않으므로 baseline 비교의 공정성에는 영향이 없다.
  - 로컬 timm 1.0.31에 `vit_small_patch16_dinov3`가 있어 설치 문제는 적어 보인다(미실행).
- 정성 비교 그림: 같은 실패 이미지에서 Lite-Mono와 Depth Pro 2의 경계를 나란히 보여준다. "상한선(upper bound) 참고"로만 쓴다.

## 8. Depth Pro 2로 Eigen test를 다시 돌린 결과 (2026-11-27)

실행: `analysis/run_depth_pro_2.py`(분리된 환경 `third_party/.venv-dp2`). 원본 해상도로 넣고, 초점거리는 KITTI 보정값을 썼다. 평가는 Lite-Mono와 **같은 코드**를 썼다(median scaling, 80m, Garg crop). 비교는 `analysis/compare_models.py`, 결과는 `outputs/depth_pro_2/compare_*/compare.md`.

Depth Pro 2는 KITTI로 학습하지 않은 zero-shot 지도학습 모델이라 baseline 경쟁 상대가 아니다. **"어떤 실패가 모델로 고쳐지고, 어떤 실패는 정답(LiDAR) 문제인가"를 가르는 기준선**으로만 쓴다.

### Depth Pro 2-S (24.3M, 0.01 s/img, 0.5 GB)

| 항목 | Lite-Mono | DP2-S |
|---|---:|---:|
| 전체 AbsRel / δ₁ | 0.107 / 0.887 | 0.108 / 0.890 |
| LM 최악 30% 209장의 AbsRel | 0.158 | 0.139 |
| 40–80m AbsRel (log 치우침) | 0.201 (−0.147) | 0.189 (−0.075) |
| 경계 위치 정확(±4px) | 25.4% | 42.1% |
| 배경 쪽 번짐 >4px | 27.7% | 15.0% |
| 앞 물체 놓침 | 22.9% | 22.5% |
| 배경에 끝까지 안 닿음 | 16.9% | 12.9% |

- metric(스케일 보정 없음): AbsRel 0.129, δ₁ 0.847.
- LM 최악 30% 209장 중 **131장은 DP2-S도 최악 30%**다. 덤불·철망(0086), 트럭 정체(0052), 유리 진열창 벽(549), 검은 차(68)처럼 장면이나 GT 문제로 보인다.
- **LM에서만 실패하는 이미지는 23장**이고 0047·0029(고속도로·트인 도로)에 몰려 있다. 먼 가로수·수풀 띠를 너무 멀게, 옆 비탈을 너무 가깝게 예측하는 등 장면 전체의 깊이 배치(전역 구조)가 틀린다.

### 개선 방향에 주는 시사점

1. **경계 번짐은 모델로 고칠 수 있다.** DP2-S에서 번짐 비율이 절반 아래로 준다(27.7% → 15.0%). 원본 해상도 처리와 학습형 upsampling이 원인과 맞닿아 있다. A안(decoder 고해상도 경로)의 근거다.
2. **"앞 물체 놓침" 약 22%는 두 모델이 같다.** 모델 문제라기보다 LiDAR와 카메라의 위치 차이나 GT 오류일 가능성이 높다. 경계 지표에서 이 부분은 개선 대상에서 빼고, 해석할 때 한계로 적는다.
3. **트인 장면의 전역 구조 오류가 LM 고유 실패의 주 원인이다.** 경계와는 별개 문제라 A안으로는 기대하기 어렵다. 후보: 저해상도 단계의 전역 문맥 강화(LGFI 위치·개수). 범위를 넘으면 한계로만 적는다.
4. **덤불·철망·유리처럼 GT가 모호한 장면은** 큰 모델도 나쁘다. 보조 평가로 KITTI improved GT(Uhrig et al. 2017, eigen_benchmark 652장)를 쓰는 것을 검토한다(데이터 추가 다운로드 필요).

### Depth Pro 2-H+ (872M, 0.117 s/img, 5.5 GB) 추가

| 항목 | Lite-Mono | DP2-S | DP2-H+ |
|---|---:|---:|---:|
| 전체 AbsRel / δ₁ | 0.107 / 0.887 | 0.108 / 0.890 | **0.074 / 0.945** |
| LM 최악 30%의 AbsRel | 0.158 | 0.139 | 0.100 |
| 평탄 영역 AbsRel | 0.086 | 0.089 | 0.055 |
| 40–80m AbsRel (log 치우침) | 0.201 (−0.147) | 0.189 (−0.075) | 0.136 (−0.037) |
| 윤곽 바깥 배경 쪽 AbsRel | 0.288 | 0.266 | 0.240 |
| 투과 점(유리·철망) AbsRel | 0.357 | 0.359 | **0.355** |
| 경계 위치 정확(±4px) | 25.4% | 42.1% | 51.1% |
| 배경 쪽 번짐 >4px | 27.7% | 15.0% | 14.1% |
| 앞 물체 놓침 | 22.9% | 22.5% | 19.5% |
| 배경에 끝까지 안 닿음 | 16.9% | 12.9% | 8.2% |

- metric(스케일 보정 없음): AbsRel 0.076, δ₁ 0.943. KITTI zero-shot인데도 Lite-Mono보다 크게 좋다. 학습 데이터 19종에 Waymo·VKITTI2 같은 주행 데이터가 포함되어 있어 공정한 비교 대상은 아니다.
- **투과 점의 AbsRel은 세 모델 모두 약 0.355로 같다.** 모델 크기와 무관한 GT 문제임이 확인된다.
- **"앞 물체 놓침"은 H+에서도 19.5%가 남는다.** 대부분 GT 쪽(LiDAR와 카메라 시점 차이) 원인으로 본다.
- 번짐(>4px)은 S와 H+가 비슷하다(15.0%, 14.1%). 번짐 감소는 모델 크기보다 **구조(원본 해상도, 학습형 upsampling)** 효과로 해석할 수 있다. 24M인 S도 이미 절반으로 줄인다. → A안의 근거를 강화한다.
- H+ 기준으로 LM에서만 실패하는 이미지는 25장이다(0117, 0029, 0047, 0036). 트인 장면과 먼 구조물 중심이다.
