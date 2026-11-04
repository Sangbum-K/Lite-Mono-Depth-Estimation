# Configs

실험마다 달라지는 설정과 서버 경로를 분리한다.

- `paths.env`: 현재 서버의 데이터·가중치·출력 절대경로. Git 제외.
- `paths.env.example`: 변수 이름만 제공하는 복사본. Git 포함.
- Phase 2 이후 baseline과 개선안의 실제 CLI 설정을 별도 파일로 저장한다.

아직 공식 코드의 config loader를 만들지 않았다. 존재하지 않는 YAML 실행 기능을 가정하지 않고, 실제 실행 가능한 CLI와 함께 설정 방식을 확정한다.
