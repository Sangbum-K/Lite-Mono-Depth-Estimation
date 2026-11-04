# Environment

실제 서버의 CUDA·Python 환경을 재현하기 위한 폴더다.

Phase 1에서 다음 파일을 추가한다.

- `requirements-lock.txt` 또는 `environment.yml`: 실제 설치에 성공한 버전.
- `server_info.md`: GPU, VRAM, driver, CUDA, OS, Python, PyTorch 정보.
- 필요하면 `compatibility_notes.md`: 공식 Lite-Mono 코드의 호환 수정과 이유.

설치 전 예상 버전을 적지 않고, 실제 서버에서 동작을 확인한 버전만 고정한다.
