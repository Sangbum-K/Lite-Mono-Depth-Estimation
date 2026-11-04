# Lite-Mono 선행연구 및 실험 설계

조사 기준일: 2026-11-02, Asia/Seoul.

**2026-11-03 최종 확정:** Lite-Mono를 baseline으로 사용한다. 12GB VRAM 기준의 [구현 계획](../PLAN.md)을 현재 실행 기준으로 삼는다.

용도: Lite-Mono 논문 이해와 연구 기획을 위한 메모. 제출 보고서 원고가 아니다. 아래 제안은 실험 전 가설이며, 성능 향상이나 독창성이 아직 입증되었다는 뜻은 아니다.

과제 근거: [Assignment 2 안내](../HW2.pdf), 3쪽 전체 재확인. 연구 근거는 논문 원문·학회 proceedings·저자 공식 코드다. 이번 조사에서는 학습·추론을 실행하지 않았고 가중치 다운로드 성공도 확인하지 못했다.

## 1. 판단부터 정리

| 질문 | 판단 | 핵심 이유 |
|---|---|---|
| 현재 SOTA인가? | 현재 최고 성능이라고 소개하면 부정확 | 발표 당시에도 절대 정확도보다 모델 크기 대비 성능이 강점. 이후 같은 Lite-Mono 기반 개선 결과가 발표됨 |
| 과제 baseline으로 써도 되는가? | 명시된 요건에 부합 | CVPR 2023 논문이며 KITTI가 원문 benchmark. 과제는 최신 SOTA가 아닌, 지정 학회 논문의 재현과 자체 baseline 대비 개선을 요구 |
| 지금도 관련성이 있는가? | 있음 | ECCV 2024 후속 연구가 직접 Lite-Mono를 사용하고, 2025–2026에도 경량 깊이 추정 연구가 계속됨 |
| 개선 여지가 남았는가? | 문헌상 분명히 있음 | 같은 3.1M 단일 프레임 추론 모델에서도 학습 방식 개선 결과가 존재. 단, 이것이 학생의 특정 아이디어가 성공한다는 뜻은 아님 |
| 독창적 개선이 쉬운가? | 쉬운 편은 아님 | 경계·다중 스케일·일관성·증류·증강은 이미 연구가 많음. 기존 방법 적용과 자신이 바꾼 부분을 명확히 구별해야 함 |
| 최종 선택 | Lite-Mono로 확정 | 12GB 단일 GPU에서 공식 평가를 먼저 재현하고 변경 범위를 한 모듈로 제한 |

권장 출발점은 **Lite-Mono 기본 모델 + 640×192 + KITTI Eigen 평가**다. 원거리·동적 객체·경계·야간을 모두 개선하는 계획은 범위가 과하다. 먼저 재현 결과에서 실제 약점 하나를 골라야 한다.

## 2. SOTA를 어떻게 해석해야 하는가

Lite-Mono는 단일 이미지에서 깊이를 예측하는 경량 네트워크다. KITTI 단안 영상으로 자기지도 학습하고, 학습 중에는 깊이 네트워크 외에 카메라 자세를 추정하는 PoseNet도 사용한다. 따라서 3.1M이라는 추론 모델 크기가 전체 학습 비용을 뜻하지는 않는다. [원문](https://arxiv.org/pdf/2211.13202), [학습 코드](https://raw.githubusercontent.com/noahzn/Lite-Mono/main/trainer.py)

비교할 때 다음을 고정하거나 차이를 밝혀야 한다.

1. 추론 입력이 한 장인가, 여러 프레임인가.
2. 깊이 GT 없이 학습하는가, GT·스테레오·의미 라벨·큰 교사 모델을 쓰는가.
3. ImageNet·외부 영상 등 사전학습 정보가 같은가.
4. KITTI raw Eigen GT인지, improved GT인지. crop·depth cap·scale alignment는 같은가.
5. 입력 해상도·파라미터·연산량·실제 지연시간이 어느 정도인가.

단안 깊이 추정 전체를 하나의 순위로 묶으면 이런 조건이 섞인다. 특히 Depth Anything의 KITTI metric fine-tuning 결과나 다중 프레임 추론 결과를 Lite-Mono와 숫자만 비교하면 연구 설정의 차이를 놓친다.

### 비교 가능한 설정을 우선한 문헌 수치

아래는 대체로 640×192, KITTI raw Eigen, 단일 프레임 추론의 공개 보고값이다. 서로 다른 저자의 수치이며 이번 조사에서 직접 재현한 결과가 아니다. 학습 정보·예산까지 완전히 통제된 단일 실험표로 해석하면 안 된다.

| 방법 | 추론 모델 크기 | AbsRel ↓ | RMSE ↓ | δ₁ ↑ | 해석 |
|---|---:|---:|---:|---:|---|
| Monodepth2-Res18 | 14.3M | 0.115 | 4.863 | 0.877 | Lite-Mono가 채택한 자기지도 학습 방식의 중요한 출발점 |
| Lite-Mono | 3.1M | 0.107 | 4.561 | 0.886 | 작은 모델 크기와 정확도의 균형이 강점 |
| MonoViT-tiny | 10.3M | 0.102 | 4.459 | 0.895 | Lite-Mono 원문 표에도 더 높은 정확도의 큰 모델이 존재 |
| Mono-ViFI + Lite-Mono, 단일 프레임 | 3.1M | 0.103 | 4.349 | 0.891 | 추론 모델 크기를 유지하면서 학습 프레임워크를 개선 |
| NimbleD + Lite-Mono | 3.1M | 0.096 | 4.304 | 0.903 | 큰 교사 모델의 pseudo-label과 추가 영상 사전학습 사용 |
| RTS-Mono-XS, 2025 preprint | 2.4M | 0.101 | 4.357 | 0.897 | 최근 경량 연구 사례. 확인한 버전은 preprint여서 게재 확정 논문과 구별 |

앞의 세 행은 [Lite-Mono 원문 Table 2](https://arxiv.org/pdf/2211.13202), Mono-ViFI 행은 [ECCV 2024 Table 1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06123.pdf), NimbleD 행은 [저자 결과표](https://github.com/xapaxca/nimbled), RTS-Mono 행은 [2025 preprint Table II](https://arxiv.org/pdf/2511.14107)에 근거한다. 파라미터 집계 방식은 논문별 차이가 있을 수 있다.

이 표로부터 얻을 수 있는 결론은 다음과 같다.

- Lite-Mono를 현재의 절대 정확도 SOTA라고 소개할 근거는 없다. 이 조사는 2026년의 모든 논문을 동일 환경에서 비교한 전수 leaderboard 검증도 아니다.
- 발표 당시에도 더 큰 MonoViT-tiny의 정확도가 높았으므로, “모든 모델 중 정확도 1위”가 핵심 기여는 아니었다.
- Mono-ViFI의 Lite-Mono 단일 프레임 결과는 AbsRel 0.107→0.103, 약 3.7% 상대 오차 감소다. 개선 가능성의 직접적인 문헌 근거다. 다만 학습에는 VFI와 일관성 학습 등 추가 비용이 든다.
- NimbleD의 0.107→0.096은 약 10.3% 상대 오차 감소지만, 외부 데이터·교사 모델을 쓴 결과다. 이를 순수한 작은 모듈의 효과처럼 해석하면 안 된다.

## 3. 읽어야 할 선행연구와 연결 관계

| 연구 | 학회·상태 | 무엇을 알아야 하나 | Lite-Mono 프로젝트와의 관계 |
|---|---|---|---|
| Monodepth2: Digging Into Self-Supervised Monocular Depth Estimation | ICCV 2019 | minimum reprojection, auto-masking, multi-scale 학습 | Lite-Mono의 손실·학습 구조를 이해하기 위한 필수 배경 |
| HR-Depth | AAAI 2021 | 큰 깊이 gradient 영역의 오류, 고해상도 특징·skip connection·feature fusion | 경계 개선을 decoder 수정으로 접근할 때 우선 읽을 연구 |
| RA-Depth | ECCV 2022 | scale augmentation, 해상도 변화, cross-scale depth consistency | crop/resize와 일관성 학습은 이미 있는 방향임을 확인 |
| MonoViT | 3DV 2022 | CNN과 Transformer를 이용한 지역·전역 특징 | Lite-Mono의 효율성 기여와 정확도 tradeoff를 이해. 3DV는 이번 과제의 baseline 학회 목록에는 없음 |
| Mono-ViFI | ECCV 2024 | VFI를 통한 시간 증강, 단일·다중 프레임 학습, depth consistency | Lite-Mono를 그대로 활용한 직접적인 후속 개선 근거. 가장 먼저 볼 후속 연구 |
| NimbleD | ECCV 2024 CV4Metaverse Workshop | 큰 모델 pseudo-label, 자기지도 손실 결합, 대규모 영상 사전학습 | 증류를 통한 개선 가능성과 외부 정보 비용. workshop을 ECCV 본학회로 표기하면 안 됨 |
| RoboDepth | NeurIPS 2023 Datasets and Benchmarks | 날씨·조명·센서·처리 오류에 대한 깊이 모델 강건성 | clean KITTI의 좋은 성능과 실제 교란 강건성은 다른 문제임을 이해 |

원문·공식 구현:

- [Monodepth2 저자 코드](https://github.com/nianticlabs/monodepth2)
- [HR-Depth AAAI 원문](https://cdn.aaai.org/ojs/16329/16329-13-19823-1-2-20210518.pdf), [공식 코드](https://github.com/shawLyu/HR-Depth)
- [RA-Depth ECCV proceedings](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4496_ECCV_2022_paper.php), [공식 코드](https://github.com/hmhemu/RA-Depth)
- [MonoViT 원문·게재 정보](https://arxiv.org/abs/2208.03543)
- [Mono-ViFI ECCV proceedings](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6123_ECCV_2024_paper.php), [공식 코드](https://github.com/LiuJF1226/Mono-ViFI)
- [NimbleD 원문](https://arxiv.org/abs/2408.14177), [공식 코드](https://github.com/xapaxca/nimbled)
- [RoboDepth NeurIPS proceedings](https://papers.neurips.cc/paper_files/paper/2023/hash/43119db5d59f07cc08fca7ba6820179a-Abstract-Datasets_and_Benchmarks.html), [공식 코드](https://github.com/worldbench/RoboDepth)

추천 읽기 순서는 **Monodepth2의 학습 원리 → Lite-Mono → HR-Depth와 Mono-ViFI → 선택한 개선 방향의 추가 논문**이다. 관련연구를 모두 구현할 필요는 없다.

### 2025–2026 동향에서 확인되는 점

RTS-Mono는 경량 encoder와 sparse feature fusion을 연구하며, 공식 저장소는 확인 시점에 논문을 submission의 early version으로 설명한다. 따라서 최신 경향 참고로 사용하고, 과제의 적격 baseline 게재 근거로 사용하지 않는다. [공식 README](https://github.com/ZYCheng777/RTS-Mono)

2026년 Sensors 논문 *Lightweight Monocular Depth Estimation with Local Feature Enhancement Modules and Guided Data Augmentation*은 Lite-Mono를 기반으로 구조와 학습 증강을 변경한다. 이는 Lite-Mono가 여전히 연구 출발점으로 쓰인다는 근거다. 다만 이 논문은 from-scratch 100 epoch 등 별도 프로토콜을 사용하므로 원본 ImageNet-pretrained 30 epoch 결과와 그대로 순위를 매기면 안 된다. 구조 축소·crop·disparity-guided cutout도 이미 연구된 방향임을 보여준다. [2026-08-21 게재 원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC13517643/)

## 4. baseline으로 사용할 수 있는가

### 과제 규정에 대한 적합성

Assignment 2는 지정 학회 논문을 baseline으로 삼고 그 논문이 사용한 benchmark 하나를 재현하도록 요구한다. Lite-Mono는 CVPR 2023이고 KITTI를 사용하므로 조건에 부합한다. 예시 목록 밖 주제도 허용되며, 최신 SOTA여야 한다는 요구는 없다. 후속 연구 적용도 인용하면 허용되고, 직접 재현한 baseline 대비 성능 향상이 필수다. [과제 안내](../HW2.pdf)

따라서 “2023년 모델이라 쓰면 안 된다”는 문제가 아니라, 재현 조건을 명확히 하고 개선의 효과를 제대로 분리할 수 있는지가 핵심이다.

### 공식 코드 재현성에 대한 실제 확인 범위

| 항목 | 확인 결과·주의점 |
|---|---|
| 모델·학습·평가 코드 | 공개되어 있음. 다만 이번 조사에서 설치·실행하지 않음 |
| 가중치 | README에 KITTI 모델과 ImageNet encoder 링크가 있음. SURFdrive 링크를 조사 도구로 열지 못했으므로 실제 다운로드 가능 여부는 미확인. 링크가 죽었다고 단정할 근거도 없음 |
| 기본 학습 split | options는 `eigen_zhou`, trainer는 해당 폴더의 train/val 목록을 읽음. 확인한 저장소 root에는 `splits/eigen`만 표시되고 기본 train/val raw URL은 404였으므로, Monodepth2 공식 목록 보완 여부를 먼저 확인해야 함 |
| 학습 epoch | README 실행 예시는 30, options 기본값은 50. 실제 실행값을 명시해야 함 |
| 장비 | 기본 trainer는 CUDA, 평가도 `.cuda()`를 직접 호출. 현재 Mac 환경에서 수정 없이 돌아간다고 가정하면 안 됨 |
| 라이브러리 | dataset에 `Image.ANTIALIAS`, decoder에 `timm.models.layers` 등 구형 코드 관용구가 있어 버전 고정·호환성 확인 필요 |

근거: [README](https://github.com/noahzn/Lite-Mono), [options](https://raw.githubusercontent.com/noahzn/Lite-Mono/main/options.py), [trainer](https://raw.githubusercontent.com/noahzn/Lite-Mono/main/trainer.py), [dataset](https://raw.githubusercontent.com/noahzn/Lite-Mono/main/datasets/mono_dataset.py), [decoder](https://raw.githubusercontent.com/noahzn/Lite-Mono/main/networks/depth_decoder.py), [평가 코드](https://raw.githubusercontent.com/noahzn/Lite-Mono/main/evaluate_depth.py).

이는 baseline 부적격 사유가 아니라 재현 준비 작업이다. 환경·데이터 목록을 보완할 때 평가 정의와 모델 동작을 조용히 바꾸지 말고, 수정 내용과 근거를 남겨야 한다.

### 데이터 수와 평가 프로토콜

원문은 학습 triplet 39,180개를 기재한다. Monodepth2가 공개한 `eigen_zhou/train_files.txt`는 39,810행이고 val은 4,424행이며, Lite-Mono의 Eigen test 목록은 697행이다. 원문 숫자와 표준 파일 수가 다르므로 실제 채택한 목록·출처·유효 sample 수를 기록해야 한다. 원문 오타인지 추가 필터링인지 이번 조사만으로 확정하지 않는다. [공식 학습 목록](https://raw.githubusercontent.com/nianticlabs/monodepth2/master/splits/eigen_zhou/train_files.txt), [검증 목록](https://raw.githubusercontent.com/nianticlabs/monodepth2/master/splits/eigen_zhou/val_files.txt), [Lite-Mono test 목록](https://raw.githubusercontent.com/noahzn/Lite-Mono/main/splits/eigen/test_files.txt)

평가는 80m cap, 지정 crop, valid mask, 이미지별 median scaling을 유지한다. improved GT에서 얻은 숫자와 raw Eigen 숫자를 섞지 않는다. 이미지별 GT median을 이용해 scale을 정렬하는 평가는 실제 차량에서 미터 단위 절대 거리를 바로 얻는 성능과 다르다. [공식 평가 코드](https://raw.githubusercontent.com/noahzn/Lite-Mono/main/evaluate_depth.py)

데이터를 축소해야 한다면 train 목록에서 정당한 규칙·비율을 정하고 각 sample의 이전·현재·다음 프레임을 함께 확보한다. test697을 보고 모델을 고르지 않는다. full raw 다운로드 안내는 약175GB이므로, 데이터·압축 해제·전처리·로그를 위한 저장공간도 학습 계획에 포함해야 한다. 필요한 목록에 맞춰 데이터를 준비하는 경우 실제 용량은 달라질 수 있다. [원본 준비 안내](https://github.com/nianticlabs/monodepth2#-kitti-training-data)

## 5. 개선 여지와 독창성은 구별해야 한다

후속 논문이 좋아진 결과를 보였으므로 **개선 가능한 공간 자체는 존재한다.** 그러나 작은 개선 모듈이 항상 좋아지거나, 이미 알려진 모듈 조합이 곧 새로운 연구가 되는 것은 아니다.

Lite-Mono에는 이미 다음이 있다.

- 지역·전역 특징 결합과 multi-scale 예측.
- image edge를 고려한 disparity smoothness.
- minimum reprojection과 auto-masking.
- 밝기·색·대비·flip 계열 증강.

따라서 “edge loss, attention, multi-scale, 밝기 증강을 처음 도입했다”는 주장은 피해야 한다. [손실 구현](https://raw.githubusercontent.com/noahzn/Lite-Mono/main/layers.py), [학습 코드](https://raw.githubusercontent.com/noahzn/Lite-Mono/main/trainer.py), [데이터 증강](https://raw.githubusercontent.com/noahzn/Lite-Mono/main/datasets/mono_dataset.py)

### 후보별 현실성

| 방향 | 구현 난도·비용에 대한 판단 | 개선 실험의 형태 | 주의할 점 |
|---|---|---|---|
| decoder의 제한된 feature fusion 변경 | 중간. teacher·추가 데이터 없이 시작 가능 | 최종 upsampling 주변의 한 지점에 경량 gate/residual을 두고 기존 concat과 비교 | HR-Depth·RTS-Mono 등과 중복이 큼. 더 큰 모델이어서 좋아진 효과를 분리 |
| crop/scale 일관성 학습 | 중상. 추가 forward로 학습 비용 증가 | 기존 자기지도 손실에 정렬된 두 view의 일관성 항을 추가 | RA-Depth·Mono-ViFI를 인용. frame들의 동일 변환과 intrinsics·scale 처리가 틀리면 기하학이 깨짐 |
| 큰 모델의 pseudo-depth 증류 | 중상. 교사 추론·캐시·외부 사전학습 정보 비용 | 일반 증류와 신뢰도·영역 가중 증류를 비교 | NimbleD가 이미 직접적인 선행연구. 상대 depth/disparity 정렬 없이 미터 단위 L1을 걸면 부적절 |
| 원거리 전용 손실·가중치 | 중상~높음 | 원거리 오차가 실제 약점인지 거리 구간별로 먼저 분석 | 자기지도 학습에는 GT 거리 라벨이 없고, 예측 기반 가중치는 불안정할 수 있음. 먼 영역만 개선되고 전체가 악화될 수 있음 |
| 날씨·영상 교란 강건성 | 구현은 비교적 단순할 수 있으나 검증 범위 증가 | clean 평가와 교란 평가를 함께 보고 강건성 비교 | 강건성만 좋아지고 원래 benchmark 성능이 떨어지면 과제의 개선 요구를 충족한다고 단정하기 어려움 |

이 표의 난도·우선순위는 프로젝트 설계 판단이며, 각 제안에 대한 성능 측정 결과는 아니다.

## 6. 특히 '차량 경계·원거리'를 주제로 삼을 때의 함정

### 경계는 보기 쉽지만 정확히 평가하기가 간단하지 않다

KITTI raw Eigen의 LiDAR 기반 GT는 희소하다. 경계 주변에는 깊이 점이 적거나 불확실할 수 있다. 따라서 다음을 피해야 한다.

- 비어 있는 depth 픽셀을 0m 정답으로 보고 경계를 추출하기.
- RGB 영상의 색·그림자·차선 표시 경계를 모두 실제 깊이 경계라고 가정하기.
- 교사 모델의 pseudo-depth와의 일치도를 GT 정확도 향상으로 소개하기.
- 깊이 그림이 더 날카롭다는 이유만으로 실제 오차가 줄었다고 결론 내리기.

경계 부근 정량 분석은 valid GT와 고정된 영역 정의를 사용하고, 유효 픽셀 수를 함께 보여야 한다. 차량만 분석하려면 별도 객체·영역 주석의 출처와 benchmark frame 대응도 필요하다. 초기에는 전체 깊이 오차를 주 지표로 두고, 경계 시각화는 실패 분석의 증거로 활용하는 편이 현실적이다. sparse GT와 improved GT 차이는 [Mono-ViFI의 평가 설명](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06123.pdf)에서도 구분된다.

### 원거리 개선을 첫 번째 유일 목표로 잡지는 않는 편이 좋다

원거리는 영상상 시차·물체 크기가 작고 정보가 부족해, 단순한 loss 변경으로 해결하기 어렵다. 우선 GT로 정의한 고정 거리 구간(예: 0–20m, 20–40m, 40–80m)에서 어느 구간의 오차가 큰지 분석한다. 실제 구현은 공식 valid range와 동일하게 제한한다.

scale alignment는 전체 공식 valid mask에서 이미지별 한 번 수행한 후 거리별로 나눠야 한다. 구간마다 별도 median scaling을 하면 그 구간의 scale 오차를 인위적으로 제거해 결과를 좋게 만들 수 있다. 비교할 양쪽 모델은 같은 구간 정의와 표본을 사용한다.

앞선 '경계와 원거리 개선' 추천은 연구 관심사의 예시였다. **경계·원거리 두 문제를 동시에 해결하는 계획은 이번 과제의 첫 범위로 과하다.**

## 7. 가장 권하는 연구 구성

권장 연구 질문:

> Lite-Mono의 encoder와 자기지도 학습 방식을 유지하면서, decoder의 제한된 특징 결합 변경으로 경계 주변의 깊이 번짐을 줄이고 전체 KITTI 깊이 오차도 낮출 수 있는가?

이 방향을 우선하는 이유는 외부 데이터·큰 teacher·복잡한 VFI 모듈을 동시에 도입하지 않아 비교 원인을 분리하기 쉽기 때문이다. 성공 가능성이 입증되었다는 뜻은 아니다. decoder 교체 자체의 전 세계적 독창성도 주장하지 않는다.

공식 decoder는 upsampling, skip feature concat, convolution, bilinear disparity upsampling으로 구성된다. 한 위치만 수정하는 실험 지점이 비교적 명확하다. [decoder 코드](https://raw.githubusercontent.com/noahzn/Lite-Mono/main/networks/depth_decoder.py)

### 권장 대조 실험

| 실험 | 목적 |
|---|---|
| B: 원본 Lite-Mono | 원래 설정의 재현 기준 |
| B + 단순 추가 convolution | 추가 파라미터·학습 용량의 효과를 확인하는 대조군 |
| B + 문헌 기반 fusion | HR-Depth 등의 적용 효과를 확인하고 출처 명시 |
| B + 제안 gate/residual | 자신이 바꾼 부분이 단순 용량 증가나 기존 기법보다 유용한지 확인 |

처음에는 B와 후보 하나만 짧게 비교하고, 방향이 보일 때 대조군을 확장한다. 외부 정보가 없는 작은 개선이 실패하면, 다음 후보로 학습 시에만 쓰는 일관성 제약이나 인용한 증류 방법을 검토할 수 있다.

주 지표는 **전체 AbsRel**, 보조로 나머지 표준 깊이 지표·파라미터·latency·메모리를 보고한다. 경계 사례·거리 구간 분석은 왜 변화했는지 설명하는 근거다. 주 지표를 결과를 본 뒤 유리한 것으로 바꾸지 않는다.

## 8. 공정한 재현·개선의 최소 조건

1. 같은 train/val/test 목록, 같은 데이터 규모와 전처리.
2. 같은 초기 가중치·입력 해상도·PoseNet·optimizer·학습 예산. 다른 부분은 모두 명시.
3. 공개 checkpoint를 개선안만 더 학습한다면, 원본 모델도 같은 checkpoint에서 같은 예산으로 추가 학습하는 대조군을 둠.
4. 방법·threshold·epoch 선택은 val로 수행. held-out test는 최종 비교에 사용.
5. crop·80m cap·median scaling·post-processing·GT 종류를 동일하게 유지.
6. 작은 향상은 seed 변동인지 확인. 자원이 허락하면 최종 후보와 baseline을 여러 seed로 비교.
7. 데이터 축소는 정당한 기준·비율·근거와 실제 자원 사용량을 기록. 일부러 약하게 학습한 baseline을 만들지 않음.
8. 특정 경계 그림만 개선되어도 전체 지표가 악화되면 성공으로 단정하지 않음. 실패한 시도도 남김.

이런 기준으로 baseline을 고정하면, 최신 SOTA 전체를 넘어야 한다는 부담 없이 연구의 원인·결과를 설명할 수 있다. 과제의 필수 비교 상대는 자신이 충실하게 재현한 baseline이다.

## 9. 자원·일정과 최종 판단

원문은 단일 TITAN Xp, batch12로 학습하고 ImageNet 초기화 모델에는 30 epoch를 사용한다. 35 epoch 학습은 약15시간으로 보고한다. 이것은 저자의 장비·환경 결과이며 현재 장비에서 측정한 시간이 아니다. 한 번 학습 가능한 것과 baseline·개선·대조군을 여러 번 학습할 수 있는 것은 다르다. [원문 §4.2](https://arxiv.org/pdf/2211.13202)

현재 날짜에서 Proposal(11/15), Intermediate(11/29), Final(12/18)까지 각각 13일, 27일, 46일이다. 다음은 일정 배치 제안이다.

- 11/02–11/05: 가중치·데이터 목록 확보, 환경 고정, 공식 평가 또는 작은 평가 smoke run.
- 11/06–11/09: 짧은 학습으로 메모리·시간 측정, 실제 가능한 재현 예산 확정.
- 11/10–11/15: baseline 실패 사례와 관련연구에 근거해 연구 질문·재현 계획 확정. 제출문은 본인 문장으로 작성.
- 11/16–11/29: baseline 정량 결과, 후보 한 가지의 초기 결과 확보.
- 11/30–12/11: 핵심 개선·대조 실험과 최종 후보 확인.
- 12/12–12/18: 고정 평가, 결과 분석, 코드·환경·보고 정리.

첫 단계에서 (a) 가중치 또는 대체 초기화 경로가 확보되는지, (b) 공식 평가 정의를 재현할 수 있는지, (c) 같은 조건의 학습을 여러 번 감당할 수 있는지 확인해야 한다. VRAM 상한은 12GB이며 GPU 모델·사용 시간은 아직 미확정이므로 학습 규모와 완주 가능성은 서버에서 측정한 뒤 확정한다.

**추천:** Lite-Mono는 이번 과제에 적합한 연구 baseline이다. 최신 SOTA라는 설명은 빼고, '경량 자기지도 단안 깊이 추정의 재현과 제한된 decoder 개선'으로 범위를 잡는 것이 좋다. 작은 변경을 공정하게 비교하고 실패 이유까지 설명하는 구성은 과제에 맞는다. 전혀 새로운 경계·원거리 해결법을 처음부터 만들겠다는 목표는 현재 단계에서 권하지 않는다.
