# Data

KITTI 원본과 전처리 결과는 서버에만 두고 Git에는 올리지 않는다.

- `raw/`: 다운로드한 KITTI raw data.
- `processed/`: 변환 결과와 재생성 가능한 cache.
- `splits/`: 실제 사용한 train/val/test 목록과 출처 기록. Git에 포함.

Phase 2에서 다음을 기록한다.

- 다운로드 출처와 이용 조건
- 실제 디렉터리 구조와 총 용량
- split 출처 commit, 행 수, 파일 누락 수, 해시
- JPG/PNG 선택과 전처리
- 학습에 사용한 RGB·인접 프레임과 평가에만 사용한 LiDAR GT의 구분

원본 데이터 파일을 이 폴더에 넣어도 `.gitignore`가 차단한다. split 목록은 재현에 필요하므로 `splits/`에 보관한다.
