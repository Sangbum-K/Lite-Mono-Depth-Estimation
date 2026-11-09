# Environment

학습 환경은 Colab Pro다. GPU 종류(T4/L4/A100)는 세션마다 배정되므로 run마다 기록한다.

- `requirements-colab.txt`: Colab 기본 설치(torch, torchvision) 위에 추가하는 패키지. scheduler 패키지는 commit을 고정했다.
- 실제 버전 기록: 노트북 A 셀이 세션마다 `nvidia-smi`, CPU 수, 디스크, `pip freeze`를 Drive의 `lite_mono/env/session_*.txt`에 저장한다. 보고서에 쓸 세션 기록은 이 폴더로 복사한다.
- 공식 코드의 호환 수정은 [vendor/README.md](../vendor/README.md)에 있다.

Colab 제약 (2026-11-08 공식 FAQ 기준, 변할 수 있음):

- Pro 세션은 보통 최대 12시간, 24시간 연속 실행은 Pro+만 해당. 브라우저를 닫거나 오래 비우면 끊길 수 있다.
- GPU 사용량은 compute unit으로 차감된다. 차감 속도는 GPU 종류마다 다르며 실제 값은 Colab 화면에서 확인한다.
- 그래서 학습은 매 epoch Drive에 저장하고 이어서 학습할 수 있게 했다.

설치 전 예상 버전을 적지 않고, 실제로 동작한 버전만 기록한다.
