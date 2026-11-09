# Improving Lite-Mono for Monocular Depth Estimation

**확정 주제:** 경량 자기지도 단안 깊이 추정. Lite-Mono(CVPR 2023)를 KITTI에서 재현하고, 실제 실패 분석에 근거한 작은 decoder 개선을 검증한다.

학습 환경은 **Colab Pro**다(GPU는 세션마다 배정, run마다 기록). 기본 Lite-Mono, 640×192 입력을 유지하고, baseline과 개선안을 동일 학습·평가 조건으로 비교한다. 전체 AbsRel이 주 지표이며 효율과 다른 깊이 지표도 함께 확인한다.

현재 완료한 일은 주제 선정, 단계별 계획, 공식 코드·split·가중치 확보, 호환 수정, Colab 실행 준비(2026-11-08), KITTI 준비와 Drive 업로드, Colab smoke test 통과(2026-11-09)다. 공개 checkpoint 재현(0.107)과 학습은 아직이다.

**시작점:** Colab에서 [scripts/colab_lite_mono.ipynb](scripts/colab_lite_mono.ipynb)를 연다 ([`Colab` 브랜치 노트북 바로 열기](https://colab.research.google.com/github/Sangbum-K/Lite-Mono-Depth-Estimation/blob/Colab/scripts/colab_lite_mono.ipynb)). 실행용 코드는 GitHub `Colab` 브랜치에 올린다. **무엇이 어디 있고 Colab에서 학습이 어떻게 돌아가는지는 [GUIDE.md](GUIDE.md)에 쉽게 정리했다.** 지금까지 바꾼 것의 쉬운 요약은 [CHANGES.md](CHANGES.md), 공식 코드 수정 근거는 [vendor/README.md](vendor/README.md)와 [vendor/changes_from_upstream.patch](vendor/changes_from_upstream.patch)에 있다.

- [전체 구현 계획과 Phase별 완료 기준](PLAN.md)
- [폴더 구조와 서버 작업 방법](STRUCTURE.md)
- [과제 안내](HW2.pdf)
- [Lite-Mono 선행연구 및 실험 설계](research/lite_mono_review.md)

| 제출 | 마감 시각, Asia/Seoul |
|---|---|
| Proposal | 2026-11-15 23:59 |
| Intermediate | 2026-11-29 23:59 |
| Final PDF + 코드 zip | 2026-12-18 23:59 |

다음 작업: KITTI는 PC에서 준비를 마쳤다(`data/processed/` 4개 파일 → Drive `MyDrive/lite_mono/data/`). Colab 노트북을 1 → 2 → 3 → 4 → 5(공개 checkpoint 0.107 확인) 순서로 실행한 뒤 6단계 baseline 학습을 시작한다. Colab 경로는 노트북 1단계에서 정하므로 `configs/paths.env`는 쓰지 않는다.

연구 메모와 구현 계획은 학습·작업 관리용이다. 제출 보고서는 과제 지침에 따라 학생이 자신의 이해와 문장으로 작성한다.
