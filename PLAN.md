# Assignment 2: Lite-Mono 구현 계획

확정일: 2026-11-03. 최종 정리: 2026-11-04. 시간대: Asia/Seoul.

상태: **주제와 단계별 계획 확정. 구현·환경 설치·데이터 확보·GPU 실험은 아직 시작하지 않았다.**

이 파일은 구현과 실험을 관리하는 작업 문서다. 제출 보고서 원고가 아니다. 보고서는 과제 안내에 따라 학생이 이해한 내용을 자신의 문장으로 작성한다.

## 1. 프로젝트 범위

| 항목 | 확정 내용 |
|---|---|
| 문제 | 경량 자기지도 단안 깊이 추정 |
| baseline | Lite-Mono, CVPR 2023, 기본 약 3.1M depth network |
| benchmark | KITTI raw 기반 학습, 원문의 Eigen 평가 |
| 입력 | 추론 RGB 한 장, 640×192. 학습은 인접 프레임과 PoseNet 포함 |
| 자원 | 단일 NVIDIA CUDA GPU, VRAM 상한 12GB를 계획 기준으로 사용 |
| 주 지표 | 전체 평가 이미지의 AbsRel ↓ |
| 함께 보고할 지표 | SqRel, RMSE, RMSE log, δ₁/δ₂/δ₃, 파라미터, batch 1 지연시간, peak memory |
| 우선 연구 방향 | 실제 실패를 확인한 뒤, decoder 특징 결합 한 부분을 바꿔 경계 주변 깊이 번짐을 줄이는 가설 검증 |
| 최종 성공 조건 | 동일 split·평가·학습 예산에서 직접 재현한 baseline보다 AbsRel 개선. 다른 지표의 변화와 비용도 공개 |

깊이 추정 자체가 이번 과제의 컴퓨터비전 문제다. 이면도로 안전·잠재 위험 예측은 장기 응용 동기로 설명한다. 이번 결과만으로 위험 예측 성능 향상까지 입증했다고 주장하지 않는다.

첫 구현에서는 encoder, PoseNet, 입력 해상도, 기본 자기지도 손실을 유지한다. 고해상도 확대, 큰 teacher, 외부 영상 추가, VFI·optical-flow 네트워크 결합은 범위에 넣지 않는다. 구체적인 개선 모듈은 Phase 4의 실패 분석 이후 확정한다.

## 2. 마감일과 전체 단계

날짜는 작업 목표다. GPU 기종·실제 사용 가능 시간과 데이터 확보 속도가 아직 확인되지 않았으므로, Phase 2에서 측정한 실행 비용으로 조정한다. 과제 마감 시각은 각각 23:59다.

| Phase | 목표 기간 | 핵심 작업 | 완료 증거 | 현재 상태 |
|---|---|---|---|---|
| 0 | 11/03–11/04 | 범위 확정·논문 이해·평가 규칙 초안 | 이 계획, 논문 구조·손실 설명, 프로토콜 초안 | 주제·계획 완료, 학습 노트 진행 필요 |
| 1 | 11/04–11/06 | 공식 코드·환경·가중치 확보 | 고정 commit, 환경 기록, checkpoint 로드 성공 | 미착수 |
| 2 | 11/05–11/09 | KITTI·split·평가·12GB 동작 검증 | 데이터 점검 결과, checkpoint 평가, GPU profile | 미착수 |
| 3 | 11/09–11/18 | 원본 baseline 재학습 | 전체 학습 로그, checkpoint, 정량·정성 결과 | 미착수 |
| 4 | 11/18–11/22 | baseline 실패 분석·개선 가설 선택 | validation 오류 분석, 모듈 설계 1개 | 미착수 |
| 5 | 11/22–11/27 | 개선 모듈 구현·짧은 대조 실험 | 동일 예산 pilot, 모듈 동작·메모리 확인 | 미착수 |
| 6 | 11/28–12/10 | 개선안 전체 학습·ablation | B/I/C 비교표, 실패한 실험 기록 | 미착수 |
| 7 | 12/10–12/14 | 최종 검증·반복 실행·효율 측정 | 설정을 고정한 benchmark 결과, 속도·자원 기록 | 미착수 |
| 8 | 12/14–12/17 | 재실행 점검·결과 정리·제출 패키징 | 실행 안내, 최종 PDF, 코드 zip 점검 | 미착수 |

의존 관계: Phase 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. Phase 2의 다운로드·정리는 Phase 1과 병행할 수 있고, 보고서에 필요한 사실·실험 기록은 전 단계에서 계속 쌓는다.

| 제출 | 마감 | 그때 확보해야 할 내용 | 내부 점검 목표 |
|---|---|---|---|
| Proposal | 11/15 23:59 | 문제·데이터 규모/출처/전처리/split, baseline·관련연구 이해, 재현 계획. 전체 학습이 진행 중이면 실제 상태를 기재 | 11/13 자료 고정, 11/14 본인 작성·검토 |
| Intermediate | 11/29 23:59 | Proposal 요약 갱신, 직접 재현 결과, 실패 분석, 개선안 및 초기 결과 | 11/27 표·그림 확보, 11/28 본인 작성·검토 |
| Final | 12/18 23:59 | 전체 이야기, 동일 조건의 성능 향상, 시도·실패·ablation, 근거·남은 한계, 코드 zip | 12/17 패키지 점검 완료 |

## 3. Phase별 구현 내용과 완료 기준

### Phase 0. 문제와 프로토콜 정의

작업:

- Lite-Mono의 CDC·LGFI, depth decoder, PoseNet, view synthesis를 연결해 설명한다.
- photometric loss, minimum reprojection, auto-masking, edge-aware smoothness가 이미 baseline에 포함되어 있음을 정리한다.
- Monodepth2 → Lite-Mono → HR-Depth → Mono-ViFI 순서로 필요한 부분을 읽는다. 후속 논문 전체를 재구현하지 않는다.
- 아래 4절의 데이터·평가·비교 규칙을 프로토콜 초안으로 삼고, Phase 2에서 실제 manifest와 실행 설정으로 고정한다.

완료 기준: RGB → depth/Pose → 재투영 → loss → 학습의 흐름, 단안 scale ambiguity, median scaling의 의미를 자신의 말로 설명할 수 있다. '경계 개선'은 검증할 가설이며 확정된 새 기여가 아니다.

### Phase 1. 공식 코드와 실행 환경

작업:

- 저자 저장소를 `vendor/Lite-Mono/`에 확보하고 원본 commit과 라이선스를 기록한다. 공식 코드와 프로젝트 변경 diff를 추적할 수 있게 한다.
- CUDA GPU의 모델·VRAM·드라이버, Python·PyTorch·torchvision·timm·scheduler 버전을 기록한다. 현재 Mac 환경에서 CUDA 학습을 했다고 기록하지 않는다.
- 기본 Lite-Mono의 **ImageNet encoder 초기화**와 **KITTI 학습 완료 depth checkpoint**를 구별해서 확보하고 checksum을 기록한다.
- 구형 Pillow API, timm import, scheduler 의존성, iterator API 등의 호환성을 확인한다. 필요한 수정은 환경 호환 수정으로 기록하고 모델 개선과 구별한다.
- 모델 로드와 RGB 한 장 추론을 확인한다. 깊이 network뿐 아니라 PoseNet을 포함한 학습 경로도 확인한다.

완료 기준: 동일 환경을 재구성할 설치 기록이 있고, checkpoint 로드에 예상치 못한 missing/unexpected key가 없으며, 기본 추론이 끝난다. 가중치 링크의 실제 다운로드 성공은 이 단계에서 검증한다.

### Phase 2. 데이터·평가·메모리 확인

작업:

- KITTI 이용 조건·출처를 기록하고 RGB, 인접 프레임, calibration, 평가용 LiDAR/GT를 준비한다. 학습 loss에 depth GT를 넣지는 않는다.
- Lite-Mono가 참조하는 `eigen_zhou` train/val 목록이 선택한 commit에 있는지 확인한다. 누락됐다면 공식 데이터 안내가 연결한 Monodepth2의 목록을 출처·commit과 함께 확보한다.
- 참고 규모는 Monodepth2 목록의 train 39,810 / val 4,424, Eigen test 697이다. **실제 확보한 목록의 행 수·유효 파일 수를 직접 확인한 뒤 확정**한다. 원문과 숫자가 다르면 조용히 맞추지 않고 차이를 기록한다.
- 현재 프레임과 t−1/t+1, 카메라 side, calibration, LiDAR 파일의 누락·중복·split 교집합을 검사한다. 드라이브·인접 프레임 공유 여부도 확인하고 원본 split의 특성을 기록한다.
- GT 생성·정렬과 crop·scale alignment를 검증한다. 소수 이미지 실행은 smoke test로만 표기한다.
- 공개 KITTI checkpoint로 공식 Eigen 평가를 재현한다. 공개 보고값 AbsRel 0.107은 평가 점검의 참고값이며, 실제 재학습 결과와 같다고 가정하지 않는다.
- 고정 validation 목록 전체를 평가할 경로를 준비한다. LiDAR GT를 사용한다면 평가·모델 선택에만 사용하고 학습 loss에는 넣지 않는다.
- 공식 trainer의 `val()`은 단일 minibatch 점검이고 depth metric도 batch 전체에서 scale을 계산하는 근사치다. 이 값을 per-image scaling을 쓰는 공식 evaluator의 전체 점수로 대체하지 않는다.
- batch 6으로 시작해 forward·backward·optimizer step을 실행하고 batch 12도 측정한다. validation·checkpoint 저장까지 포함해 메모리를 확인한다.

완료 기준:

- 데이터 목록·출처·개수·해시와 missing file 점검 결과가 남아 있다.
- 공개 checkpoint 평가가 재실행 가능하며, 참고값과 차이가 있으면 원인을 설명한다. 원인을 모른 채 개선 실험으로 넘어가지 않는다.
- 약 100 step의 준비 점검과 가능하면 전체 1 epoch를 통해 peak allocated/reserved memory, GPU 프로세스 사용량, epoch 시간을 기록한다.
- 전체 학습을 12GB 안에서 수행할 설정과 baseline·개선 반복에 필요한 시간을 산출한다.

### Phase 3. baseline 재학습

작업:

- 공개 ImageNet 초기화에서 원본 Lite-Mono를 학습한다. 공개 KITTI checkpoint 평가만 한 것을 직접 재학습으로 부르지 않는다.
- 공식 README의 사전학습 기준 30 epochs를 목표로 하며, `batch_size`, 해상도, learning rate, scheduler 등을 명시적으로 저장한다. options 기본값을 그대로 사용했다고 가정하지 않는다.
- seed, 데이터 순서, augmentation, PoseNet 초기화, 학습 코드 revision을 기록한다.
- loss/learning rate, validation 지표, 메모리, 실제 GPU 시간과 wall time을 남긴다.
- 원본 마지막 epoch checkpoint를 보존한다. 주 비교 checkpoint는 사전에 정한 동일 validation 선택 규칙으로 결정한다.
- 고정 validation에서 정량 결과와 사례를 확보한다. 중간보고에 공식 benchmark 재현 수치가 필요하면 Eigen 결과도 기록하되 이후 방법 선택은 validation으로 수행한다.

완료 기준: 직접 학습한 baseline checkpoint와 정확한 실행 조건이 있고, 7개 depth 지표·대표 예측·자원 사용량을 다시 만들 수 있다. 논문 수치와 차이가 나면 데이터·초기화·batch·평가 변경 및 seed 차이를 분석한다. 일부러 약하게 학습한 모델을 개선의 기준으로 삼지 않는다.

### Phase 4. 실패 분석과 개선안 선택

작업:

- 우선 고정 validation에서 전체 오차 분포, 깊이 번짐, 작은/먼 구조물, 동적 객체 주변 등의 사례를 분석한다.
- 거리 구간은 예를 들어 0–20m, 20–40m, 40–80m로 사전 고정할 수 있다. 전체 공식 valid mask에서 이미지별 scaling을 한 번 수행한 뒤 나눈다. 구간별 scaling은 하지 않는다.
- KITTI LiDAR가 희소하므로 RGB edge를 실제 depth boundary GT처럼 취급하지 않는다. 경계 정량화는 GT 유효성·표본 수를 먼저 검증하고, 불안정하면 정성 분석으로 한정한다.
- 성공·실패 사례를 같은 고정 목록에서 함께 보여주고, 전체 AbsRel과 연결되는 실패 유형 하나를 고른다.
- HR-Depth 등 기존 decoder fusion 연구와 차이를 정리한다. 적용한 기존 방법과 자신이 추가한 부분을 구별한다.
- 모듈 위치, feature shape, 추가 파라미터, 예상 학습/추론 비용, 검증할 주장 하나를 설계 메모에 적는다.

완료 기준: '무엇이 실패했는가 → 어떤 변경이 왜 도움이 될 수 있는가 → 어떤 비교로 확인할 것인가'가 이어지는 가설 하나가 있다. 경계·원거리·동적 객체를 동시에 해결하는 설계로 확대하지 않는다.

### Phase 5. 개선 모듈과 pilot

작업:

- decoder의 한 특징 결합 지점에 작은 residual/gating 등 후보 하나를 구현한다. 구체 구조는 Phase 4 이후 확정한다.
- 기존 설정에서 모듈을 끄면 원본 출력·파라미터 로딩을 복구할 수 있게 한다. 텐서 shape, forward/backward, NaN/Inf, gradient 흐름을 점검한다.
- baseline과 개선안을 같은 초기화·데이터·짧은 epoch 예산으로 비교한다. 예: 각각 5 epochs. pilot 점수를 30 epoch baseline과 비교해 개선이라고 주장하지 않는다.
- 개선 모듈을 포함해 12GB 내 동작하는지 다시 측정한다. 모듈 때문에 batch를 바꾸면 baseline도 같은 최종 batch로 다시 맞춘다.
- 짧은 결과로 코드를 점검하고 가능성 없는 후보를 제외한다. 최종 성능의 근거는 Phase 6의 동일 전체 예산 실험으로 남긴다.

완료 기준: 모듈 동작 점검 결과, 초기 정량/정성 결과, 메모리 기록이 있고 중간보고에서 방향과 한계를 설명할 수 있다. 성능이 나빠진 결과도 보존한다.

### Phase 6. 전체 비교와 ablation

| ID | 실험 | 목적 | 예산 계획 |
|---|---|---|---|
| B | 원본 Lite-Mono | 직접 재현한 비교 기준 | 30 epochs, Phase 3 결과 재사용 가능 |
| I | B + 선택한 개선 모듈 | 제안 변경의 효과 | B와 동일 초기화·split·seed·학습 길이 |
| C | B + 단순 convolution 또는 핵심 기작 제거 버전 | 단순 용량 증가와 제안 기작의 효과 구별 | 가능한 한 I와 비슷한 추가 파라미터, 동일 학습 예산 |

작업:

- 기본 실행 예산은 B/I/C 총 3회의 전체 학습을 목표로 계산한다. C는 선택한 가설을 가장 직접적으로 검증하는 하나로 제한한다.
- validation으로 방법·hyperparameter·checkpoint를 선택한다. 외부 정보·augmentation·loss를 바꾸면 별도 변경으로 기록한다.
- 모든 실험의 전체 지표와 비용을 동일 포맷으로 수집한다. 유리한 사례·지표만 골라 결론을 만들지 않는다.
- 12/10까지 구조와 설정을 고정한다. 결과를 보고 주 지표를 바꾸지 않는다.

완료 기준: B/I/C 비교로 어떤 부분이 효과가 있는지 설명할 수 있다. 개선이 없으면 구현 오류·학습 수렴·가설을 확인하고, 문헌에 근거한 제한된 대안을 같은 조건에서 검증한다. 실패를 감추거나 개선을 보장하지 않는다.

### Phase 7. 최종 검증과 효율

작업:

- 고정한 B/I 설정을 동일 Eigen benchmark와 동일 지표로 비교한다.
- 가능하면 두 번째 seed에서 B/I를 쌍으로 추가 학습한다. 추가 비용은 30 epochs × 2회다. 한 seed만 있으면 작은 차이의 안정성이 검증되지 않았음을 명시한다.
- 동일 하드웨어·소프트웨어·정밀도·입력에서 depth encoder+decoder의 batch 1 latency를 측정한다. 준비 실행 후 GPU synchronization을 포함하고, 전처리 포함 여부와 반복 횟수를 기록한다.
- 학습 메모리와 추론 메모리, depth-only 추론 비용과 PoseNet 포함 학습 비용을 구별한다.
- 공통 색 범위의 RGB/GT/B/I/오차 그림과 개선·악화 사례를 고정 표본에서 생성한다.

완료 기준: 전체 AbsRel 개선 여부를 실제 수치로 판단할 수 있고, 나머지 지표·효율·남은 실패를 함께 설명한다. 가장 좋은 seed 하나만 뽑아 평균 성능처럼 제시하지 않는다.

### Phase 8. 재실행과 제출

작업:

- 설정 파일, 원본 commit, patch, 환경 기록, 데이터 준비, 학습, 평가 명령을 실제 파일과 일치시킨다.
- 데이터·가중치 절대 경로를 사용자별 설정으로 분리한다. 존재하지 않는 script를 실행 안내에 적지 않는다.
- 초기 상태에서 최소 설치·작은 실행·평가 결과 수집을 다시 확인한다. 전체 학습의 무의미한 재실행보다 실제 누락·환경 문제를 점검한다.
- 실험 표와 raw metric·checkpoint ID·그림이 서로 일치하는지 확인한다.
- 최종 PDF는 학생이 직접 작성한다. AI는 논문 이해·코딩 보조 범위로 사용한다.
- 코드 zip에 원본 코드와 변경분, 설정, 실행 안내를 포함한다. 데이터와 대형 checkpoint는 기본적으로 제외하고, 별도 허용·제출 요구가 있으면 그 기준을 따른다. 원 코드 라이선스·인용을 보존한다.

완료 기준: PDF와 zip이 실제 열리고, README의 실행 경로가 존재하며, 최종 결과를 재현할 입력·설정·코드가 명확하다. LMS 제출 완료 여부는 별도로 확인한다.

## 4. 모든 실험에 공통인 규칙

### 학습·메모리

- 원문 batch 12를 우선 목표로 하되, 실제 개선안까지 측정해 공통 batch를 확정한다. 6은 최초 점검값이며 확정된 최대 batch가 아니다.
- FP32로 먼저 동작을 확인한다. AMP가 필요하면 B/I/C 모두 동일 적용하고 수치 안정성을 검사한다.
- 운영 목표는 GPU 프로세스 메모리 약 10–11GB 이하로 여유를 확보하는 것이다. 실제 예상 사용량이나 보장치가 아니다.
- gradient accumulation이 BatchNorm을 포함한 원문 batch와 완전히 같다고 주장하지 않는다.
- 학습 depth range 옵션과 평가 depth cap을 구별한다. 원본 학습 기본 범위를 평가의 80m에 맞추겠다며 임의로 변경하지 않는다.
- VRAM 때문에 데이터 수부터 줄이지 않는다. 필요하면 먼저 batch·정밀도·데이터 읽기 방식을 조정한다.

### 데이터·평가

- benchmark는 KITTI로 유지한다. 데이터 축소가 불가피하면 기준·비율·split·이유와 실제 사용 자원을 기록하고 모든 비교에 같은 목록을 적용한다.
- 원본 GT 종류를 고정한다. raw Eigen GT와 improved GT의 결과를 섞지 않는다.
- 공식 evaluator의 valid depth 범위, 80m cap, crop, disparity resize, per-image median scaling을 B/I/C에 동일 적용한다.
- post-processing은 기본 비교에서 끄는 계획이며, 선택한 공식 참고값의 조건과 일치하는지 Phase 2에서 확인한다. 켠 추가 결과는 별도 표기한다.
- checkpoint 선택은 고정 validation의 per-image AbsRel로 수행한다. validation 전체를 평가하기 어렵다면 사전에 고정한 동일 목록을 쓰고 규모를 보고한다. trainer의 단일 minibatch score로 대체하지 않는다.
- 공식 checkpoint 평가와 중간보고에서 Eigen 결과를 미리 보았다면, 이를 완전히 관찰하지 않은 최종 test라고 소개하지 않는다. 이후 실험 선택은 validation으로 제한하고 사용 이력을 공개한다.
- 거리 구간과 사례는 보조 분석이다. 전체 AbsRel이 악화됐는데 일부 경계 그림만 좋아진 결과를 최종 성공으로 판단하지 않는다.

## 5. 실험 기록과 시간 예산

각 run에 다음을 남긴다.

`run_id, method, code_revision, config, seed, manifest_hash, init_checksum, GPU, VRAM, batch, precision, epochs, selected_epoch, peak_allocated, peak_reserved, GPU_process_memory, train_wall_time, eval_split, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, params, latency, notes`

시간 예산은 Phase 2에서 측정한 실제 값을 사용한다.

- 기본 계획: 전체 학습 3회(B/I/C), 짧은 pilot, 평가·오류 수정 여유.
- 추가 검증: B/I의 seed 한 쌍을 추가하면 전체 학습 2회가 더 필요하다.
- 총 시간은 `각 run의 실제 epoch 시간 × epoch 수 + validation + 데이터 준비 + 재실행 여유`로 계산한다. GPU 메모리 12GB만으로 학습 시간을 예측하지 않는다.
- 원문의 35 epochs 약 15시간은 저자 환경의 참고값이다. 사용자 장비의 보장 시간이 아니다.
- 시간이 부족하면 여러 아이디어를 추가하는 대신 후보 수를 줄인다. 데이터·epoch 축소가 필요하면 이유와 공통 조건을 다시 고정하고 baseline을 불리하게 만드는 비교를 피한다.

## 6. 프로젝트 구조

아래 구조를 2026-11-04에 생성했다. 안내 파일과 템플릿만 준비된 폴더도 있으며, 실제 Lite-Mono 코드·데이터·학습 script는 아직 없다. 자세한 사용 규칙은 [STRUCTURE.md](STRUCTURE.md)를 따른다.

```text
HW2/
  README.md                 # 프로젝트 진입점
  PLAN.md                   # 단계와 완료 기준
  STRUCTURE.md              # 폴더·Git·서버 사용 규칙
  environment/              # 서버와 Python/CUDA 환경 기록
  data/                     # 서버 데이터 위치와 split 기록
  vendor/Lite-Mono/         # 공식 코드, commit 고정
  configs/                  # B/I/C 및 profile의 명시적 설정
  scripts/                  # 검증된 설치·학습·평가 명령만 배치
  experiments/              # run별 설정·revision·metric·profile 기록
  analysis/                 # 고정 validation·오류 분석 코드
  outputs/                  # 재생성 가능한 대용량 결과, Git 제외
  research/                 # 선행연구 메모
  reports/                  # Proposal·Intermediate·Final 작성 공간
```

데이터·대형 가중치의 위치는 서버 저장 공간을 확인한 뒤 `configs/paths.env`에 기록한다. 데이터, checkpoint, log, 전체 prediction은 Git과 코드 zip에 포함하지 않는다. split 목록, 실행 설정, 최종 metric, 보고서에 사용한 작은 표·그림은 재현에 필요하므로 Git에 남긴다.

## 7. 다음 착수 작업

- [x] Lite-Mono 주제 확정과 12GB 기준 범위 정의.
- [x] 과제 마감·요구사항을 단계에 연결.
- [ ] Lite-Mono 공식 코드 확보, commit·라이선스 기록.
- [ ] 실제 CUDA GPU·환경 확인, 호환 환경 고정.
- [ ] ImageNet 초기화와 KITTI checkpoint 다운로드·checksum 확인.
- [ ] 데이터 manifest 확보·실제 파일 준비, 전체 평가·profile.

## 근거

- [과제 안내](HW2.pdf): 요구사항·평가·마감일·AI 사용 범위.
- [Lite-Mono 원문](https://arxiv.org/abs/2211.13202): 모델·학습·KITTI benchmark.
- [공식 코드와 학습 안내](https://github.com/noahzn/Lite-Mono): 가중치·데이터 안내·30 epochs 학습 명령.
- [공식 options](https://github.com/noahzn/Lite-Mono/blob/main/options.py): train range·frame·batch·epoch 등 설정.
- [공식 evaluator](https://github.com/noahzn/Lite-Mono/blob/main/evaluate_depth.py): per-image scaling·mask·crop·지표.
- [공식 trainer](https://github.com/noahzn/Lite-Mono/blob/main/trainer.py): 학습과 단일 minibatch validation의 구별.
- [Monodepth2 데이터·split](https://github.com/nianticlabs/monodepth2): 공식 Lite-Mono가 연결하는 데이터 준비 경로.
- [Lite-Mono 선행연구 메모](research/lite_mono_review.md): 관련연구·재현 위험·개선 방향.
