# Experiments

run마다 `run_template.md`를 복사해 `YYYYMMDD_method_seed.md`로 저장한다. 실행 전에는 가설·설정을, 종료 후에는 결과·자원 사용량·실패를 채운다.

Git에 남길 것:

- 정확한 실행 명령과 code revision
- 데이터 split hash와 초기 가중치 checksum
- seed, batch, epoch, precision
- 전체 metric과 선택 checkpoint
- GPU 모델·peak memory·학습 시간
- 결과 해석과 다음 결정

checkpoint, TensorBoard log, 전체 prediction은 `outputs/<run_id>/`에 두고 Git에는 올리지 않는다.
