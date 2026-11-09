# Data

KITTI 원본과 전처리 결과는 서버에만 두고 Git에는 올리지 않는다.

- `raw/`: 다운로드한 KITTI raw data.
- `processed/`: 변환 결과와 재생성 가능한 cache.
- `splits/`: 실제 사용한 train/val/test 목록과 출처 기록. Git에 포함.
- `kitti_archives_to_download.txt`: KITTI raw 다운로드 URL 66개. Monodepth2 commit `19e16e1`의 `splits/kitti_archives_to_download.txt` 사본.

Phase 2에서 다음을 기록한다.

- 다운로드 출처와 이용 조건
- 실제 디렉터리 구조와 총 용량
- split 출처 commit, 행 수, 파일 누락 수, 해시
- JPG/PNG 선택과 전처리
- 학습에 사용한 RGB·인접 프레임과 평가에만 사용한 LiDAR GT의 구분

## Colab에서의 위치 (`scripts/prepare_kitti.py`)

| 경로 | 내용 | 수명 |
|---|---|---|
| `/content/kitti_data` | 학습·평가 RGB jpg (45,259장) + calib. 학습의 `--data_path` | 세션마다 Drive tar에서 복원 |
| Drive `lite_mono/data/` | `kitti_jpg.tar`, `eigen_gt_depths.npz`(공식 test GT), `test_gt.npz`, `val_gt.npz`(이미지별 평가용) | 영구 |

### PC에서 준비하고 Drive에 올리기 (2026-11-09 완료)

```bash
bash scripts/download_kitti_pc.sh
```

- 필요한 zip 60개 합계 173.1GB, 가장 큰 zip 21.2GB (2026-11-08 HEAD 요청으로 확인). zip은 하나씩 받고 지우므로 작업 중 여유 공간은 약 40GB면 된다.
- 작업 폴더: `data/raw/kitti_data`(jpg), `data/raw/kitti_velo`(val/test LiDAR, 정답 깊이 재생성용으로 PC에 보관). 둘 다 Git 제외.
- 결과: `data/processed/`의 `kitti_jpg.tar`, `eigen_gt_depths.npz`, `test_gt.npz`, `val_gt.npz` 4개를 Drive `MyDrive/lite_mono/data/`에 올린다. 원본 zip, png, LiDAR는 올리지 않는다.
- 필요한 것: 시스템 `python3`의 numpy·Pillow·six, curl, tar (이 PC에 있음). 끊기면 같은 명령을 다시 실행한다.

LiDAR를 학습 폴더에 두지 않는 이유: 데이터 로더가 LiDAR 파일을 발견하면 학습 샘플마다 정답 깊이를 계산하는데, 이 값은 로그에만 쓰이고 손실에는 쓰이지 않는다. 학습 결과는 같고 속도만 느려진다.

jpg 변환은 Monodepth2와 같은 설정(quality 92, 4:2:0)을 Pillow로 적용했다. Monodepth2는 ImageMagick을 썼다.

원본 데이터 파일을 이 폴더에 넣어도 `.gitignore`가 차단한다. split 목록은 재현에 필요하므로 `splits/`에 보관한다.
