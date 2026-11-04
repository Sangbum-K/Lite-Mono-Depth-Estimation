# Analysis

고정 validation 평가와 실패 분석 코드를 둔다.

예정 분석:

- 전체 KITTI depth metric
- 거리 구간별 오차
- baseline과 개선안의 이미지별 차이
- 개선·악화 사례 선택
- 파라미터, latency, peak memory 비교
- 보고서용 표·그림 생성

분석 기준과 표본 목록은 결과를 보기 전에 가능한 한 고정한다. 생성된 대용량 prediction은 `outputs/`에 두고, 최종 보고서에 사용한 작은 그림은 `reports/figures/`에 복사한다.
