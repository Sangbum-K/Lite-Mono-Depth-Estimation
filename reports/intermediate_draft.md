<!-- title2: 재현 및 한계 분석 -->

<!-- abstract -->
본 프로젝트는 RGB 영상 한 장에서 픽셀별 깊이를 추정하는 단안 깊이 추정 문제를 다룬다. Baseline인 Lite-Mono(CVPR 2023)를 원 논문과 같은 KITTI Eigen-Zhou split으로 학습했고, Eigen test에서 AbsRel 0.107을 얻어 논문의 결과를 재현했다. 재현한 모델의 오차를 분석한 결과, 오차는 물체와 배경이 만나는 깊이 경계에 집중되었고 물체의 깊이가 배경 쪽으로 번지는 현상이 가장 크게 나타났다. 이를 줄이기 위해 decoder에 고해상도 feature와 학습 가능한 upsampling을 추가하는 방법을 제안한다.
<!-- /abstract -->

# 1. 서론

단안 깊이 추정(monocular depth estimation)은 카메라 한 대로 촬영한 RGB 영상 한 장에서 픽셀별 깊이를 추정하는 문제다. 영상 한 장에는 거리 정보가 직접 담겨 있지 않으므로, 모델은 물체의 크기, 원근, 가림(occlusion) 같은 단서로 깊이를 추론해야 한다. 자율주행에서는 LiDAR가 정확한 거리 정보를 제공하지만 비용이 높아, 카메라 영상만으로 거리를 인식하는 방식이 함께 사용되고 있다. Ground truth 깊이를 수집하는 데에도 LiDAR가 필요하므로, 연속된 영상 프레임만으로 학습하는 자기지도(self-supervised) 방식이 널리 연구되고 있다.

Baseline으로는 경량 자기지도 모델인 Lite-Mono를 선택했다. Lite-Mono는 Monodepth2보다 파라미터 수가 약 78% 적으면서 AbsRel도 낮아, 차량처럼 연산 자원이 제한된 환경에 적합하다. 또 학습 방식은 Monodepth2를 유지하고 모델 구조만 바꾸었기 때문에, 개선안을 적용했을 때 그 효과를 다른 요인과 구분하기 쉽다. Proposal 이후 baseline을 직접 학습해 논문의 결과를 재현했고, 재현한 모델의 실패 사례를 분석해 개선할 한계를 정했다. 본 보고서는 재현 과정과 결과, 한계 분석, 그리고 이를 바탕으로 한 개선 아이디어를 다룬다.

# 2. 데이터셋

## 2.1 KITTI raw

KITTI는 차량 주행 중 수집한 데이터셋으로, 스테레오 컬러 영상과 Velodyne LiDAR 측정값, GPS/IMU 정보, 카메라 보정 정보를 포함한다. 본 프로젝트에서는 Lite-Mono 원 논문과 같이 KITTI raw를 사용한다. 라이선스는 CC BY-NC-SA 3.0으로 비상업적 연구 목적으로 사용할 수 있으며, 전체 용량은 약 175GB다.

## 2.2 데이터 분할

원 논문과 같이 학습·검증에는 Eigen-Zhou split을, 최종 평가에는 Eigen test 697장을 사용한다. Eigen-Zhou split은 Eigen split의 학습 장면에서 카메라가 거의 움직이지 않은 프레임을 제외한 뒤, 남은 프레임을 학습용과 검증용으로 나눈 것이다. 학습 샘플은 같은 카메라로 촬영한 연속된 세 프레임(t−1, t, t+1)으로 구성하고, LiDAR 깊이값은 학습에는 사용하지 않고 평가에만 사용한다.

Split 목록은 Monodepth2의 공식 파일을 그대로 사용했다. Lite-Mono 논문에는 학습 이미지 수가 39,180장으로 적혀 있지만 공식 목록은 39,810장이므로, 논문의 표기 오류로 보고 공식 목록을 따랐다. 별도의 데이터 샘플링 없이 전체 분할을 사용했다.

<!-- table widths=1650,900,2056 caption="사용한 KITTI split" -->
| Split | 이미지 수 | 용도 |
|---|---|---|
| Train (Eigen-Zhou) | 39,810 | 학습 (t−1, t, t+1) |
| Val (Eigen-Zhou) | 4,424 | 학습 중 검증 |
| Test (Eigen) | 697 | 최종 평가 (LiDAR GT) |

## 2.3 전처리 및 평가 방법

입력 영상은 640×192로 크기를 조정하고, 학습 시에는 50% 확률로 좌우 반전을 적용하며 color jitter는 세 프레임에 동일하게 적용한다. Proposal에서는 Monodepth2와 같은 방식으로 원본 PNG 영상을 JPG로 변환할 계획이었으나, 실제 학습에서는 변환하지 않고 원본 PNG를 그대로 사용했다. JPG로 학습된 공개 checkpoint를 PNG 영상으로 평가했을 때도 논문과 같은 수치가 나왔으므로, 평가에서는 영상 포맷에 따른 차이가 거의 없다. 학습에서 포맷에 따른 차이는 아직 비교하지 않았다.

평가는 원 논문과 같은 방식으로 Eigen test 697장에 대해 진행하며, LiDAR 측정값을 ground truth로 사용한다. 자기지도 방식으로 학습한 모델은 깊이의 절대적인 scale을 알 수 없으므로, 이미지마다 median scaling을 적용해 예측 깊이의 scale을 ground truth에 맞춘다. 또 80m 이내의 깊이만 사용하고, 원 논문과 같은 영상 중앙 영역(Garg crop)만 평가한다.

# 3. Baseline: Lite-Mono

자기지도 단안 깊이 추정 모델은 depth network와 pose network로 구성된다. Depth network는 입력 영상의 disparity 맵을 추정하고, pose network는 인접한 두 프레임 사이의 카메라 움직임을 추정한다. 두 추정값으로 인접 프레임에서 현재 영상을 재구성하고, 실제 영상과의 차이(photometric loss)를 줄이도록 학습하므로 ground truth 깊이가 필요 없다. 추론에는 depth network만 사용한다.

**구조.** Lite-Mono의 depth network는 encoder-decoder 구조다. Encoder는 네 단계로, 첫 단계에서 convolution으로 해상도를 절반으로 줄이고, 나머지 세 단계마다 여러 CDC 모듈과 하나의 LGFI 모듈을 순차적으로 배치해 1/4, 1/8, 1/16 해상도의 feature를 만든다. CDC는 dilation rate가 서로 다른 depth-wise convolution으로 파라미터를 늘리지 않고 receptive field를 넓히고, LGFI는 채널 간 attention으로 적은 연산량으로 global context를 반영한다. Decoder는 encoder 세 단계의 feature를 bilinear upsampling으로 키운 뒤 결합하여 disparity를 출력한다. Pose network와 학습 손실은 Monodepth2와 같다.

**선행연구 개선점.** Monodepth2(14.3M)는 convolution만 사용해 receptive field가 제한되고, Transformer 기반 모델은 정확도는 높지만 모델이 크고 느렸다. Lite-Mono는 CDC와 LGFI로 두 문제를 함께 완화해 3.1M으로 AbsRel 0.107을 기록했다.

**한계.** Proposal에서는 학습 방식이 Monodepth2와 같아 움직이는 물체에서 오차가 생길 수 있다는 점을 한계로 예상했다. 이번 분석에서는 이와 별개로, decoder 구조 때문에 깊이 경계에서 오차가 크게 나타나는 한계를 찾았다(5장).

# 4. Baseline 재현

## 4.1 학습 설정

학습은 Lite-Mono 공식 저장소의 코드와 README에 제시된 학습 명령을 따랐다. 공식 코드의 기본값은 batch 16, 50 epoch로 논문의 설정과 달라, 논문과 같은 batch 12, 30 epoch를 직접 지정했다. 전체 설정은 아래 표와 같다.

<!-- table widths=1500,3106 caption="학습 설정" -->
| 항목 | 설정 |
|---|---|
| 코드 | 공식 저장소 (commit 4874b35) |
| 입력 해상도 | 640 × 192 |
| 입력 영상 | 원본 PNG (논문: JPG) |
| Encoder 초기화 | ImageNet 사전학습 가중치 |
| Batch / Epoch | 12 / 30 |
| Optimizer | AdamW (weight decay 0.01) |
| Learning rate | Depth 1e-4→5e-6, Pose 1e-4→1e-5 (cosine) |
| Loss | Photometric(SSIM+L1), min. reprojection, auto-masking, smoothness |
| Seed | 고정하지 않음 (공식 코드와 같음) |

공식 코드는 이전 버전의 라이브러리를 기준으로 작성되어 최신 환경에서는 그대로 실행되지 않았다. 이에 따라 Pillow, torchvision, NumPy에서 삭제된 함수를 대체 함수로 바꾸었고, 학습과 평가의 계산 과정은 수정하지 않았다. 또 평가 코드에는 PNG 옵션을 주어도 항상 JPG 파일을 읽는 오류가 있어, PNG 파일을 읽도록 수정했다. 데이터를 불러오는 worker 수는 기본값 12에서 RAM이 부족해 학습이 중단되었으므로 6으로 줄였다.

## 4.2 실험 환경

실험은 로컬 서버(NVIDIA RTX 5070 12GB, Ubuntu 24.04, PyTorch 2.11/CUDA 12.8, RAM 15GB)에서 진행했다. 원 논문도 12GB GPU 한 장으로 학습했으므로, 데이터를 줄이지 않고 논문과 같은 batch 크기로 학습할 수 있었다. 학습 중 GPU 메모리는 약 8.4GB를 사용했고, 30 epoch 학습에 약 9시간(epoch당 약 18분)이 걸렸다. GPU 사용률은 약 47%에 머물렀는데, worker 수를 줄여 데이터를 불러오는 속도가 학습 속도를 따라가지 못했기 때문일 가능성이 높다. 모델의 파라미터 수는 3.07M, FLOPs는 5.03G다.

<!-- full -->
<!-- table widths=2938,950,950,950,950,950,950,1000 bold=3 caption="KITTI Eigen test 재현 결과" -->
| 방법 | AbsRel↓ | SqRel↓ | RMSE↓ | RMSE log↓ | δ1↑ | δ2↑ | δ3↑ |
|---|---|---|---|---|---|---|---|
| Lite-Mono 논문 | 0.107 | 0.765 | 4.561 | 0.183 | 0.886 | 0.963 | 0.983 |
| 공개 checkpoint (PNG 영상으로 평가) | 0.107 | 0.766 | 4.560 | 0.183 | 0.886 | 0.963 | 0.983 |
| 재현 (30 epoch) | 0.107 | 0.821 | 4.608 | 0.184 | 0.887 | 0.962 | 0.982 |

<!-- figure fig1_qualitative.png width=full -->
그림 1. 재현한 Lite-Mono의 예측 결과. 왼쪽부터 입력 영상, 예측 disparity(밝을수록 가까움), LiDAR 점에서의 AbsRel(파란색일수록 작고 빨간색일수록 크다).
<!-- /full -->

## 4.3 정량 결과

원 논문과 같이 마지막 epoch의 checkpoint로 평가했다. AbsRel, RMSE log, δ1~δ3는 논문과 0.001 이내로 일치해 baseline을 재현했다. 반면 SqRel은 0.821로 논문(0.765)보다 약 7% 높고, RMSE도 4.608로 논문(4.561)보다 조금 높다. SqRel은 오차를 제곱해 평균하므로 큰 오차에 민감한 지표로, 학습 후반의 epoch 사이에서도 0.795~0.867 범위로 변동했다. 이 때문에 이 차이가 PNG 입력에 의한 것인지, seed를 고정하지 않아 생긴 학습 간 편차인지는 아직 구분할 수 없다.

같은 평가 코드로 공개 checkpoint를 평가하면 논문 수치가 그대로 나오므로, 평가 과정의 문제는 아니다. 본 프로젝트는 이 재현 모델을 이후 실험의 baseline으로 사용한다.

## 4.4 정성 결과

그림 1은 test 영상 중 무작위로 고른 세 장의 예측 결과다. 도로, 주차된 차량, 나무처럼 장면의 전체적인 깊이 구조는 대체로 맞게 예측한다. 오차는 도로 면처럼 평평한 영역에서는 작지만, 차량과 나무의 윤곽, 먼 건물과 수풀 근처에서 크게 나타난다. 다음 장에서는 이 오차가 어디에 얼마나 집중되는지 분석한다.

# 5. 한계 분석

## 5.1 오차 분포

먼저 오차가 일부 이미지에 몰려 있는지 보기 위해 Eigen test 697장의 이미지별 AbsRel을 계산했다. 오차가 가장 큰 10%의 이미지가 전체 오차에서 차지하는 비율은 약 18%로, 특정 이미지에서만 크게 틀리기보다는 대부분의 이미지에 비슷한 오차가 있다.

다음으로 LiDAR 점 단위로 오차를 나누어 보았다. 깊이 경계는 LiDAR 점 주변 9×9 영역 안에서 깊이가 1.25배 이상 차이 나는 점으로 정의했다. 깊이 경계 근처의 점은 전체의 11.5%이지만 전체 오차의 28.1%를 차지하며, AbsRel은 0.260으로 평평한 영역(0.086)의 약 3배다. 거리별로는 40m 이상에서 AbsRel이 0.201로 가장 높지만, 해당 점은 전체의 5.7%로 적어 전체 오차에서 차지하는 비율은 10.8%다.

<!-- table widths=1406,750,850,750,850 bold=2 caption="영역별 오차 (치우침이 음수면 실제보다 가깝게 예측)" -->
| 영역 | 점 비율 | 오차 비율 | AbsRel | 치우침 |
|---|---|---|---|---|
| 평평한 영역 | 88.5% | 71.9% | 0.086 | +0.4% |
| 깊이 경계 근처 | 11.5% | 28.1% | 0.260 | −8.4% |
|   배경 쪽 | 4.9% | 13.5% | 0.293 | −29.4% |
|   물체 쪽 | 4.8% | 12.7% | 0.282 | +17.6% |
| 0–20m | 76.5% | 63.3% | 0.088 | +1.3% |
| 20–40m | 17.8% | 25.9% | 0.155 | −4.3% |
| 40–80m | 5.7% | 10.8% | 0.201 | −13.7% |

경계 근처의 점을 다시 배경 쪽과 물체 쪽으로 나누면 오차의 방향이 서로 반대로 나타난다. 배경 쪽은 실제보다 평균 29% 가깝게, 물체 쪽은 18% 멀게 예측한다. 경계를 사이에 둔 두 깊이가 섞이면서, 물체의 깊이가 배경 쪽으로 번진 것이다(edge fattening).

## 5.2 깊이 경계 번짐

그림 2는 예측 깊이에서 찾은 경계를 입력 영상 위에 표시한 것이다. 차량 지붕(1, 2), 나무 기둥(3), 보행자(5) 모두 예측 경계가 실제 윤곽보다 바깥쪽에 위치해, 물체가 실제보다 크게 예측된다. 그림 3은 보행자의 경계를 확대한 것이다. LiDAR 깊이에서는 보행자와 뒤쪽 도로의 경계에서 깊이가 바로 바뀌지만, 예측에서는 보행자의 깊이가 몇 픽셀에 걸쳐 배경으로 퍼져 경계가 흐려진다.

번짐의 정도를 수치로 나타내기 위해, 양쪽 깊이가 일정한 LiDAR 깊이 경계 3,263개에서 예측 경계와 실제 경계 사이의 거리를 측정했다. 예측 경계가 실제 경계에서 4px 이내에 있는 경우는 25.4%에 불과했고, 배경 쪽으로 4px 이상 밀린 경우가 27.7%로 물체 쪽으로 밀린 경우(7.1%)의 약 4배였다.

<!-- full -->
<!-- figure fig2_overlay.png width=7900 -->
그림 2. 깊이 경계 번짐 예시. 왼쪽부터 LiDAR 깊이, Lite-Mono 예측 깊이, 입력 영상 위에 표시한 예측 경계(빨간 선). 경계가 많이 밀린 경우 중에서 주행 영상마다 하나씩 골랐다.

<!-- figure fig3_idx529.png width=full -->
그림 3. 보행자 경계 확대. (a) 입력 영상, (b) LiDAR 깊이, (c) 예측 깊이, (d) LiDAR 점에서의 오차(빨간색은 실제보다 멀게, 파란색은 가깝게 예측), (e)~(h) 상자 영역 확대. (h)는 예측 깊이 위에 LiDAR 점을 겹쳐 표시한 것이다.
<!-- /full -->

## 5.3 원인 분석

번짐의 원인을 찾기 위해 Lite-Mono의 decoder 구조를 살펴보았다. Encoder의 첫 단계는 1/2 해상도의 feature를 만들지만, 이 feature는 다음 단계의 입력으로만 사용되고 decoder에는 전달되지 않는다. 이에 따라 decoder가 사용하는 가장 세밀한 feature는 1/4 해상도다. 또 decoder는 1/2 해상도에서 disparity를 출력한 뒤, bilinear upsampling으로 크기만 2배로 늘려 최종 결과를 만든다.

이 구조는 두 가지 이유로 경계를 흐리게 만든다. (1) 해상도: 1/4 해상도의 픽셀 하나는 원본 영상에서 약 8픽셀에 해당하므로, 이 정도의 정보로는 경계의 정확한 위치를 정하기 어렵다. (2) 보간: bilinear upsampling은 주변 픽셀 값의 가중 평균으로 새 픽셀을 채우므로, 깊이가 갑자기 바뀌는 경계에서 물체와 배경 사이의 중간 깊이가 만들어진다. 앞의 분석에서 경계가 수 픽셀씩 배경 쪽으로 밀린 것은 이 구조 때문일 가능성이 높다.

한편 LiDAR와 카메라는 차량에 설치된 위치가 달라, 경계 근처에서는 카메라에는 가려서 보이지 않는 뒤쪽 면이 LiDAR에는 측정되는 경우도 있다. 이 경우 ground truth 자체가 틀리므로, 경계 오차가 모두 모델의 문제라고 보기는 어렵다.

## 5.4 Depth Pro 2와의 비교

경계 오차 중 모델 구조로 줄일 수 있는 부분과 ground truth 문제로 생긴 부분을 구분하기 위해, 같은 test 영상에 Depth Pro 2 [7]를 적용해 비교했다. Depth Pro 2는 여러 데이터셋의 깊이 정답으로 학습한 지도학습 모델로, 영상을 원본 해상도로 처리하고 transposed convolution으로 해상도를 높여 경계를 선명하게 예측하도록 설계되었다. KITTI로 학습하지 않았고 학습 방식과 모델 크기도 Lite-Mono와 크게 달라 성능을 직접 비교할 대상은 아니며, 본 프로젝트에서는 경계 오차를 해석하는 기준으로만 사용했다. 공개된 가중치로 S(24.3M)와 H+(872M) 두 모델을 추론하고, Lite-Mono와 같은 평가 코드로 평가했다.

<!-- table widths=1906,900,900,900 bold=5 caption="Lite-Mono와 Depth Pro 2 비교" -->
| 항목 | Lite-Mono | DP2-S | DP2-H+ |
|---|---|---|---|
| 파라미터 | 3.1M | 24.3M | 872M |
| AbsRel | 0.107 | 0.108 | 0.074 |
| δ1 | 0.887 | 0.890 | 0.945 |
| 경계 위치 정확 (4px 이내) | 25.4% | 42.1% | 51.1% |
| 배경 쪽 번짐 (4px 이상) | 27.7% | 15.0% | 14.1% |
| 물체 쪽 밀림 (4px 이상) | 7.1% | 7.4% | 7.1% |
| 물체를 놓침 | 22.9% | 22.5% | 19.5% |
| 투과 점 AbsRel | 0.357 | 0.359 | 0.355 |

**구조로 줄일 수 있는 오차.** Depth Pro 2-S는 전체 AbsRel이 0.108로 Lite-Mono와 거의 같지만, 배경 쪽 번짐은 27.7%에서 15.0%로 절반 가까이 줄었다. 또 S보다 약 36배 큰 H+의 번짐도 14.1%로 S와 큰 차이가 없다. 번짐은 모델 크기보다 해상도 처리와 upsampling 방식 같은 구조의 영향을 크게 받는다.

**Ground truth 문제로 생긴 오차.** 경계에서 물체를 놓친 경우는 세 모델 모두 20% 안팎으로 비슷하다. 또 유리창이나 철망처럼 LiDAR가 물체를 통과해 뒤쪽을 측정한 점(투과 점)의 AbsRel도 세 모델 모두 약 0.355로 같다. 모델 크기와 관계없이 같은 오차가 남으므로, 이 부분은 ground truth의 문제로 보고 개선 대상에서 제외한다.

**Lite-Mono에서만 나타나는 오차.** Lite-Mono에서 오차가 큰 상위 30%(209장) 중 134장은 Depth Pro 2-H+에서도 상위 30%에 속했다(그림 4(a)). 반면 25장은 Depth Pro 2-H+에서 오차가 작았는데, 대부분 고속도로처럼 트인 장면에서 멀리 있는 나무와 수풀을 실제보다 멀게 예측한 경우였다(그림 4(b)). 이는 경계가 아니라 장면 전체의 깊이 배치가 틀린 문제로, 이번 개선의 범위에서는 제외한다.

<!-- full -->
<!-- figure fig4_depthpro2.png width=full -->
그림 4. Lite-Mono와 Depth Pro 2-H+ 비교. 오른쪽 두 열은 LiDAR 점에서의 오차(빨간색은 실제보다 멀게, 파란색은 가깝게 예측). (a) 두 모델 모두 오차가 큰 장면: 주차 차량, 유리 진열창이 있는 건물. (b) Lite-Mono에서만 오차가 큰 장면: 트인 도로의 먼 수풀과 나무.
<!-- /full -->

정리하면, Lite-Mono의 오차는 깊이 경계에 집중되어 있고, 그중 모델 구조로 줄일 수 있는 부분은 물체의 깊이가 배경으로 번지는 현상이다. 본 프로젝트는 이 번짐을 개선할 한계로 정했다.

# 6. 개선 아이디어

## 6.1 제안 방법

5장의 분석을 바탕으로, decoder의 마지막 단계만 바꾸는 방법을 제안한다.

**구조.** 두 가지를 바꾼다. (1) 1/2 해상도 skip connection: encoder 첫 단계에서 만들어지지만 사용되지 않던 1/2 해상도 feature(48채널)를 decoder 마지막 단계에 연결한다. 기존 decoder보다 2배 세밀한 feature를 사용하므로 경계 위치를 더 정확하게 정할 수 있다. (2) 학습 가능한 upsampling: 최종 disparity를 bilinear upsampling 대신 transposed convolution으로 2배 키운다. Bilinear upsampling은 항상 주변 값을 평균하지만, transposed convolution은 학습한 가중치로 값을 채우므로 경계에서 깊이를 물체와 배경 중 어느 쪽으로 정할지 학습할 수 있다. Encoder, pose network, 학습 손실과 설정은 baseline과 같다.

**선행연구와의 관계.** 학습 가능한 upsampling은 Depth Pro [8]와 Depth Pro 2 [7]의 decoder에서, 고해상도 feature를 decoder에 연결하는 방식은 HR-Depth [9]에서 가져온 아이디어다. Depth Pro 2는 bilinear upsampling 대신 transposed convolution을 사용하며, 처리 해상도가 높을수록 경계가 선명해진다고 보고했다. HR-Depth는 자기지도 단안 깊이 추정에서 고해상도 feature를 decoder에서 더 잘 활용하도록 skip connection을 다시 설계했다. 본 프로젝트에서는 실패 분석으로 경계 번짐을 주요 한계로 찾고, 그 원인을 Lite-Mono decoder의 구조와 연결해 바꿀 위치를 정했다. 또 개선 효과를 경계 위치로 직접 측정하는 평가 방법을 만들었다.

**비용.** 추가되는 파라미터는 11,848개로 전체의 0.39%이므로 Lite-Mono의 경량 구조를 유지한다. Encoder 구조가 바뀌지 않아 ImageNet 사전학습 가중치를 그대로 사용할 수 있고, 개선안을 사용하지 않으면 원래 Lite-Mono와 출력이 같도록 구현했다.

**한계.** 이 방법은 경계 번짐만을 대상으로 하므로, 5.4절의 ground truth 문제나 트인 장면에서 전체 깊이 배치가 틀리는 문제는 줄이기 어렵다. 또 1/2 해상도에서 연산이 추가되므로 추론 시간이 늘어날 수 있다.

<!-- table widths=1406,1600,1600 caption="Decoder 마지막 단계 비교" -->
| 항목 | Lite-Mono | 제안 방법 |
|---|---|---|
| 가장 세밀한 feature | 1/4 해상도 | 1/2 해상도 |
| 최종 upsampling | Bilinear | Transposed conv |
| 추가 파라미터 | – | 11,848 (0.39%) |
| Encoder·손실·학습 설정 | – | 동일 |

## 6.2 예비 실험

제안한 방법의 방향을 확인하기 위해 5 epoch 학습으로 예비 실험을 진행한다. Learning rate는 baseline과 같은 cosine schedule을 사용하므로, baseline의 5번째 epoch checkpoint와 같은 조건에서 비교할 수 있다. 비교에는 Eigen test의 기본 지표와 5.2절의 경계 위치 지표를 함께 사용한다.

<!-- table widths=1906,1350,1350 caption="예비 실험 비교 (5 epoch)" -->
| 항목 | Baseline | 제안 방법 |
|---|---|---|
| AbsRel | 0.118 | ==TBD== |
| SqRel | 0.849 | ==TBD== |
| RMSE | 4.759 | ==TBD== |
| δ1 | 0.867 | ==TBD== |
| 경계 위치 정확 (4px 이내) | 19.3% | ==TBD== |
| 배경 쪽 번짐 (4px 이상) | 28.7% | ==TBD== |
| 물체를 놓침 | 25.9% | ==TBD== |

==[예비 실험 결과가 나오면 작성: 번짐 비율과 전체 지표가 어떻게 바뀌었는지]==

## 6.3 향후 계획

예비 실험에서 번짐이 줄어드는 것을 확인하면, 30 epoch 전체 학습으로 baseline과 같은 split, 같은 지표에서 비교한다. 또 skip connection과 학습 가능한 upsampling을 각각 따로 적용해 각 변경의 효과를 나누어 분석한다. 경량 모델이 목표이므로 파라미터 수와 FLOPs뿐 아니라 추론 시간도 함께 측정한다.

# 참고문헌

<!-- refs -->
[1] C. Godard, O. Mac Aodha, M. Firman, and G. J. Brostow, "Digging Into Self-Supervised Monocular Depth Estimation," ICCV, 2019.
[2] N. Zhang, F. Nex, G. Vosselman, and N. Kerle, "Lite-Mono: A Lightweight CNN and Transformer Architecture for Self-Supervised Monocular Depth Estimation," CVPR, 2023.
[3] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, "Vision meets Robotics: The KITTI Dataset," IJRR, 2013.
[4] D. Eigen, C. Puhrsch, and R. Fergus, "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network," NeurIPS, 2014.
[5] T. Zhou, M. Brown, N. Snavely, and D. G. Lowe, "Unsupervised Learning of Depth and Ego-Motion from Video," CVPR, 2017.
[6] R. Garg, V. Kumar B.G., G. Carneiro, and I. Reid, "Unsupervised CNN for Single View Depth Estimation: Geometry to the Rescue," ECCV, 2016.
[7] P. Hu, M. Mihajlovic, L. Duong, L. Lipson, W. Dong, A. Delaunoy, S. R. Richter, and V. Koltun, "Depth Pro 2: Sharp and Temporally Consistent Metric Monocular Depth," arXiv:2611.26603, 2026.
[8] A. Bochkovskii, A. Delaunoy, H. Germain, M. Santos, Y. Zhou, S. R. Richter, and V. Koltun, "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second," ICLR, 2025.
[9] X. Lyu, L. Liu, M. Wang, X. Kong, L. Liu, Y. Liu, X. Chen, and Y. Yuan, "HR-Depth: High Resolution Self-Supervised Monocular Depth Estimation," AAAI, 2021.
<!-- /refs -->
