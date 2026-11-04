# Splits

Phase 2에서 실제 사용한 `train_files.txt`, `val_files.txt`, `test_files.txt`와 출처 정보를 저장한다.

각 목록에 대해 다음을 기록한다.

- 원본 저장소와 commit
- 원본 파일의 checksum
- 행 수와 실제 존재하는 샘플 수
- 원본에서 수정했다면 변경 이유와 규칙

split을 임의로 바꾸거나 누락 파일을 조용히 제거하지 않는다.
