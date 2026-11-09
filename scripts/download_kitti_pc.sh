#!/usr/bin/env bash
# PC에서 KITTI를 받아 Colab용 파일 4개를 만든다. 끊겨도 다시 실행하면 이어서 진행한다.
#   bash scripts/download_kitti_pc.sh
# 결과: data/processed/ 의 kitti_jpg.tar, eigen_gt_depths.npz, test_gt.npz, val_gt.npz -> Drive MyDrive/lite_mono/data/
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/raw          # Git 제외. zip(최대 21GB)은 하나씩 받고 지운다
OUT=data/processed    # Git 제외. Drive에 올릴 파일

mkdir -p "$RAW" "$OUT"
# 1. 다운로드 + 필요한 프레임만 jpg 변환 + val/test LiDAR 분리 + 누락 점검 (173GB 다운로드)
python3 scripts/prepare_kitti.py --kitti_out "$RAW/kitti_data" --velo_out "$RAW/kitti_velo" --tmp "$RAW/zip"
# 2. 평가용 정답 깊이 (LiDAR는 여기서만 쓴다)
python3 vendor/Lite-Mono/export_gt_depth.py --data_path "$RAW/kitti_velo" --split eigen
cp vendor/Lite-Mono/splits/eigen/gt_depths.npz "$OUT/eigen_gt_depths.npz"
python3 analysis/eval_per_image.py build-gt --split_file data/splits/test_files.txt --velo_root "$RAW/kitti_velo" --out "$OUT/test_gt.npz"
python3 analysis/eval_per_image.py build-gt --split_file data/splits/val_files.txt --velo_root "$RAW/kitti_velo" --out "$OUT/val_gt.npz"
# 3. jpg 묶기. 폴더 이름 kitti_data와 .done 표시를 그대로 둬야 Colab E 셀이 동작한다
tar -cf "$OUT/kitti_jpg.tar" -C "$RAW" kitti_data
du -sh "$RAW/kitti_data" "$RAW/kitti_velo" "$OUT"/*
echo "완료: $OUT 의 4개 파일을 Drive MyDrive/lite_mono/data/ 에 올리세요."
