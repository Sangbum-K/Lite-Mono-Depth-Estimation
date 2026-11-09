# Analysis

고정 validation 평가와 실패 분석 코드를 둔다.

| 파일 | 역할 |
|---|---|
| `eval_per_image.py` | 공식 `evaluate_depth.py`와 같은 계산(Eigen crop, 1e-3–80m, 이미지별 median scaling, 7개 지표)을 이미지별로 수행. 여러 checkpoint를 이미지 한 번 읽어 동시에 평가하고, 각 weights 폴더에 `per_image_<split>.csv`를 저장. 정답 깊이는 sparse로 저장해 val 4,424장도 메모리 부담 없이 평가 |
| `test_eval_per_image.py` | 위 스크립트의 sparse 저장·복원과 mask·crop·scaling 계산을 numpy로 점검. 로컬에서 통과 |

검증 기준: 공개 checkpoint의 test 평균이 공식 `evaluate_depth.py` 출력과 같아야 한다(노트북 5단계). 같지 않으면 이 스크립트로 얻은 숫자를 쓰지 않는다.

val의 오른쪽 카메라 행은 카메라 3 기준 정답 깊이로 평가한다. 공식 test는 모두 왼쪽 카메라(카메라 2)다.

예정 분석:

- 거리 구간별 오차 (이미지별 scaling을 한 번 한 뒤 구간으로 나눔)
- baseline과 개선안의 이미지별 차이
- 개선·악화 사례 선택
- 파라미터, latency, peak memory 비교
- 보고서용 표·그림 생성

분석 기준과 표본 목록은 결과를 보기 전에 가능한 한 고정한다. 생성된 대용량 prediction은 `outputs/`에 두고, 최종 보고서에 사용한 작은 그림은 `reports/figures/`에 복사한다.
