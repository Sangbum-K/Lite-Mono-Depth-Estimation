# Improving Lite-Mono for Monocular Depth Estimation

**확정 주제:** 경량 자기지도 단안 깊이 추정. Lite-Mono(CVPR 2023)를 KITTI에서 재현하고, 실제 실패 분석에 근거한 작은 decoder 개선을 검증한다.

자원 기준은 단일 CUDA GPU의 **12GB VRAM**이다. 기본 Lite-Mono, 640×192 입력을 유지하고, baseline과 개선안을 동일 학습·평가 조건으로 비교한다. 전체 AbsRel이 주 지표이며 효율과 다른 깊이 지표도 함께 확인한다.

현재 완료한 일은 주제 선정과 단계별 계획이다. 공식 코드 확보·설치·데이터 다운로드·실제 GPU 실행은 아직 하지 않았다.

- [전체 구현 계획과 Phase별 완료 기준](PLAN.md)
- [폴더 구조와 서버 작업 방법](STRUCTURE.md)
- [과제 안내](HW2.pdf)
- [Lite-Mono 선행연구 및 실험 설계](research/lite_mono_review.md)

| 제출 | 마감 시각, Asia/Seoul |
|---|---|
| Proposal | 2026-11-15 23:59 |
| Intermediate | 2026-11-29 23:59 |
| Final PDF + 코드 zip | 2026-12-18 23:59 |

다음 작업은 서버에서 공식 코드·가중치·CUDA 환경을 확보하는 Phase 1이다. 먼저 `configs/paths.env.example`을 복사해 서버 경로를 기록한다. 이후 실제 실행 명령이 생기면 `scripts/`와 이 문서를 함께 갱신한다. 아직 실행 가능한 학습 프로젝트가 구성된 상태는 아니다.

연구 메모와 구현 계획은 학습·작업 관리용이다. 제출 보고서는 과제 지침에 따라 학생이 자신의 이해와 문장으로 작성한다.
