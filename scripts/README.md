# Scripts

입력 경로는 하드코딩하지 않고 CLI 인자로 받는다.

| 파일 | 역할 | 검증 상태 (2026-11-08) |
|---|---|---|
| `colab_lite_mono.ipynb` | Colab 전체 흐름: 설정 → 가중치 → smoke test → KITTI 준비 → 공개 checkpoint 평가 → 학습(이어하기) → checkpoint 선택 | JSON·bash 문법만 확인. Colab 미실행 |
| `download_kitti_pc.sh` | PC에서 KITTI 준비 전체를 실행하고 Drive에 올릴 파일 4개를 `data/processed/`에 만듦 | 문법만 확인 |
| `prepare_kitti.py` | KITTI zip을 하나씩 받아 필요한 프레임만 jpg로 변환, val/test LiDAR와 calib은 별도 폴더에 저장, 마지막에 누락 점검 | 가짜 zip으로 로컬 시험 통과 (선별·LiDAR 분리·재실행·누락 감지) |
| `smoke_test.py` | 데이터 없이 GPU에서 코드-가중치 일치, 파라미터 수, 호환 수정, forward/backward 확인 | Colab 미실행 |

노트북은 `scripts/` 밖의 [`analysis/eval_per_image.py`](../analysis/eval_per_image.py)도 사용한다. Colab에서 실제로 실행한 뒤 이 표의 검증 상태를 갱신한다.
