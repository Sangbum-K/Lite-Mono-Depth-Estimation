# 프로젝트 폴더 구조와 서버 사용법

## 핵심 원칙

1. **Git에 저장:** 코드, 설정, split 목록, 실험 기록, 최종 metric, 보고서 원고.
2. **서버에만 저장:** KITTI 원본, 사전학습 가중치, checkpoint, TensorBoard log, 전체 prediction.
3. **한 run은 한 기록:** 실행 전에 run ID와 설정을 만들고, 종료 후 결과와 자원 사용량을 같은 기록에 추가한다.
4. **원본과 수정본 분리:** 저자 코드는 `vendor/Lite-Mono/`에 두고 upstream commit을 기록한다. 환경 호환 수정과 연구 아이디어 수정을 구별한다.
5. **절대경로를 코드에 쓰지 않기:** 서버 경로는 Git에서 제외되는 `configs/paths.env`에만 기록한다.

## 권장 구조

```text
HW2/
├── README.md
├── PLAN.md
├── STRUCTURE.md
├── HW2.pdf
├── environment/
│   ├── README.md
│   └── server_info.example.md
├── configs/
│   ├── README.md
│   └── paths.env.example
├── data/
│   ├── README.md
│   ├── raw/                 # KITTI 원본, Git 제외
│   ├── processed/           # 변환·cache, Git 제외
│   └── splits/              # 실제 사용한 txt 목록, Git 포함
├── vendor/
│   ├── README.md
│   └── Lite-Mono/           # Phase 1에서 공식 코드 배치
├── scripts/
│   └── README.md
├── experiments/
│   ├── README.md
│   └── run_template.md
├── analysis/
│   └── README.md
├── outputs/                 # prediction·임시 그림, Git 제외
├── research/
│   ├── README.md
│   ├── lite_mono_review.md
│   └── paper_note_template.md
└── reports/
    ├── README.md
    ├── proposal.md
    ├── intermediate.md
    ├── final.md
    └── figures/             # 최종 보고서에 실제 사용한 작은 그림
```

빈 폴더는 GitHub에 표시되지 않는다. 서버에서 데이터와 결과가 생기면 자동으로 만들어지므로 억지로 `.gitkeep`을 넣지 않는다.

## 서버로 옮긴 직후

```bash
git clone -b Colab https://github.com/Sangbum-K/Lite-Mono-Depth-Estimation.git
cd Lite-Mono-Depth-Estimation
cp configs/paths.env.example configs/paths.env
```

그다음 `configs/paths.env`에 서버의 실제 경로만 입력한다. 이 파일은 `.gitignore`에 포함되어 있으므로 개인 서버 경로나 계정 정보가 GitHub에 올라가지 않는다.

공식 코드를 확보할 때는 먼저 upstream commit을 확인한다. 작업 방식은 Phase 1에서 서버 환경을 확인한 뒤 결정하지만, 최종 제출 zip이 혼자서도 열리도록 submodule만 남기는 방식은 피한다. 공식 코드를 복사해 사용할 경우 원 라이선스와 commit을 `vendor/README.md`에 기록한다.

## 데이터 정리

KITTI 원본을 저장소 안에 직접 복사할 수는 있지만 Git에는 올리지 않는다. 서버 용량이나 공용 데이터 위치 때문에 저장소 밖에 둔다면 symbolic link 또는 `KITTI_ROOT` 경로를 사용한다.

```text
data/raw/kitti/
├── 2011_09_26/
├── 2011_09_28/
├── 2011_09_29/
├── 2011_09_30/
└── 2011_10_03/
```

`data/splits/`에는 실제 사용한 train/val/test 파일 목록을 복사하고 출처 commit·행 수·해시를 기록한다. 원본 데이터는 없어도 split과 설정만으로 어떤 샘플을 사용했는지 알 수 있어야 한다.

## 실험 이름 규칙

run ID는 다음 형식으로 통일한다.

```text
YYYYMMDD_method_seed
```

예시:

```text
20261109_baseline_s42
20261123_gate_pilot_s42
20261201_gate_full_s42
```

각 run을 시작하기 전에 `experiments/run_template.md`를 복사해 기록한다. checkpoint와 log는 서버의 `outputs/<run_id>/`에 두고, Git에는 run 기록과 최종 metric만 남긴다.

## 파일 이름 규칙

- 공백 대신 `_`를 사용한다.
- `final_final2` 같은 이름을 만들지 않는다. 날짜나 run ID를 쓴다.
- baseline은 `baseline`, 개선안은 방법 이름, 대조군은 제거한 기능이 드러나는 이름을 쓴다.
- 보고서 그림은 `fig01_architecture.png`, `fig02_failure_cases.png`처럼 번호와 내용을 함께 쓴다.
- metric 표에는 checkpoint의 run ID를 반드시 적는다.

## 백업 기준

| 항목 | GitHub | 서버 | 별도 백업 권장 |
|---|---:|---:|---:|
| 코드·설정·split | ✓ | ✓ | 선택 |
| 실험 기록·최종 metric | ✓ | ✓ | 선택 |
| 보고서 원고·최종 그림 | ✓ | ✓ | ✓ |
| KITTI 원본 |  | ✓ | 다시 받을 수 있으면 불필요 |
| pretrained weights |  | ✓ | 링크·checksum 기록 |
| best checkpoint |  | ✓ | ✓ |
| 중간 checkpoint·log |  | ✓ | 필요할 때만 |

서버의 checkpoint는 지워질 수 있으므로 최종 baseline과 best improvement checkpoint는 다른 저장장치에도 한 번 보관한다.
