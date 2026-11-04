# Scripts

서버에서 실제로 실행해 검증한 반복 작업만 script로 만든다.

예정 역할:

- 환경·GPU 확인
- KITTI 파일과 split 검증
- baseline 학습·평가
- 개선안 학습·평가
- metric·profile 수집
- 최종 그림 생성

script를 추가할 때 입력 경로를 하드코딩하지 않고 `configs/paths.env` 또는 명시적 CLI 인자로 받는다. README에 존재하지 않는 명령을 미리 적지 않는다.
