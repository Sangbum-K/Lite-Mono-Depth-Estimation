# Scripts

입력 경로는 하드코딩하지 않고 CLI 인자로 받는다.

| 파일 | 역할 | 검증 상태 (2026-11-09) |
|---|---|---|
| `colab_lite_mono.ipynb` | Colab 전체 흐름 1~7단계: 준비 → 가중치 → smoke test → 데이터 풀기 → 공개 checkpoint 평가 → 학습(이어하기) → checkpoint 선택. 단계별 설명은 [GUIDE.md](../GUIDE.md) | 1단계는 로컬 IPython에서 가짜 Drive로 시험(clone·pull·데이터 누락 감지). 이전 버전으로 Colab 2·3단계 실행 |
| `download_kitti_pc.sh` | PC에서 KITTI 준비 전체를 실행하고 Drive에 올릴 파일 4개를 `data/processed/`에 만듦 | PC에서 실행 완료 |
| `prepare_kitti.py` | KITTI zip을 하나씩 받아 필요한 프레임만 jpg로 변환, val/test LiDAR와 calib은 별도 폴더에 저장, 마지막에 누락 점검. Colab 4단계에서는 점검만 | PC 실행 결과 RGB 45,259/45,259, LiDAR 4,884/4,884 |
| `smoke_test.py` | 데이터 없이 GPU에서 코드-가중치 일치, 파라미터 수, 호환 수정, forward/backward 확인 | Colab 통과 (파라미터 3,074,747) |

노트북은 `scripts/` 밖의 [`analysis/eval_per_image.py`](../analysis/eval_per_image.py)도 사용한다. Colab에서 새로 실행한 단계가 생기면 이 표의 검증 상태를 갱신한다.
