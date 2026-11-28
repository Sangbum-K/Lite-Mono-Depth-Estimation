#!/usr/bin/env python3
"""Progress of the 5-epoch pilots started by scripts/run_pilots.sh (read-only, no extra packages).

Usage:  scripts/pilot_status.py            # print once
        scripts/pilot_status.py --watch    # refresh every 60 s (Ctrl+C to quit)
        scripts/pilot_status.py --watch 30

Per pilot: 대기 / 학습 중 (epoch, batch, 진행률, 남은 시간, loss) / 중단됨 / 평가 중 / 분석 중 / 완료 (+ 핵심 수치).
Keep PILOTS in sync with scripts/run_pilots.sh.
"""
import csv
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
PILOTS = [("B", "20261127_smooth1e-4_e5", "smoothness 1e-4"),
          ("Alite", "20261127_hires_lite_e5", "skip8+subpixel"),
          ("C", "20261127_smoothgain10_e5", "edge gain 10"),
          ("A", "20261127_hires_decoder_e5", "skip48+deconv"),
          ("Base", "20261128_baseline_rerun_e5", "baseline 재학습"),
          ("E", "20261128_dpt24_e5", "DPT decoder dim 24")]
EPOCHS = 5
BATCHES = 39810 // 12  # Eigen-Zhou train / batch 12
LINE = re.compile(r"epoch\s+(\d+) \|.*batch\s+(\d+) \| examples/s:\s*([\d.]+) \| loss: ([\d.]+) \| "
                  r"time elapsed: (\S+) \| time left: (\S+)")


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def running(pattern):
    # list form: no shell whose own command line would match the pattern
    try:
        return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0
    except Exception:
        return False


def last_train_line(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 20000))
        tail = f.read().decode("utf8", "ignore").splitlines()
    for ln in reversed(tail):
        m = LINE.search(ln)
        if m:
            return m.groups(), os.path.getmtime(path)
    return None


def result(ana):
    try:
        rows = list(csv.DictReader(open(os.path.join(ana, "per_image.csv"))))
        absrel = sum(float(r["abs_rel"]) for r in rows) / len(rows)
        e = list(csv.DictReader(open(os.path.join(ana, "edges", "edges.csv"))))
        bleed = sum(1 for r in e if r["status"] == "located" and r["offset_px"] not in ("", "nan")
                    and float(r["offset_px"]) > 4) / len(e)
        return "AbsRel {:.3f}, 배경 쪽 번짐 {:.1%}  (baseline ep5: 0.118, 28.7%)".format(absrel, bleed)
    except Exception:
        return ""


def status():
    alive = running("bash .*run_pilots.sh")
    out = ["=== 예비실험 진행 상황  {}  (run_pilots.sh {})".format(
        time.strftime("%m-%d %H:%M:%S"), "실행 중" if alive else "꺼짐")]
    for key, run, desc in PILOTS:
        d = os.path.join(OUT, run)
        w = os.path.join(d, "models", "weights_{}".format(EPOCHS - 1), "depth.pth")
        ana = os.path.join(d, "analysis", "weights_{}".format(EPOCHS - 1))
        evlog = os.path.join(d, "eval", "weights_{}.log".format(EPOCHS - 1))
        head = "{:<6} {:<15}".format(key, desc)
        if os.path.exists(os.path.join(ana, "edges", "edges.csv")) and os.path.exists(evlog):
            out.append(head + " 완료      " + result(ana))
        elif os.path.exists(w):
            stage = "분석 중" if os.path.exists(evlog) and "abs_rel" in open(evlog).read() else "평가 중"
            out.append(head + (" " + stage if alive else " 학습 완료, 평가/분석 안 됨 (run_pilots.sh 다시 실행)"))
        elif os.path.exists(os.path.join(d, "train.log")):
            t = last_train_line(os.path.join(d, "train.log"))
            saved = len([x for x in os.listdir(os.path.join(d, "models")) if x.startswith("weights_")]) \
                if os.path.isdir(os.path.join(d, "models")) else 0
            if t is None:
                prog = "시작 중 (아직 로그 없음)"
            else:
                (ep, b, ex, loss, el, left), mt = t
                frac = (int(ep) * BATCHES + int(b)) / (EPOCHS * BATCHES)
                bar = "#" * int(frac * 20) + "-" * (20 - int(frac * 20))
                age = int(time.time() - mt)
                prog = "epoch {}/{} batch {:>4}/{} [{}] {:4.0%}  경과 {} 남음 {}  loss {}  ({}초 전 로그)".format(
                    int(ep) + 1, EPOCHS, b, BATCHES, bar, frac, el, left, loss, age)
            if running("model_name {} ".format(run)) or running("model_name {}$".format(run)):
                out.append(head + " 학습 중   " + prog)
            else:
                out.append(head + " 중단됨    " + prog + "  저장된 epoch {}개".format(saved))
        else:
            out.append(head + " 대기")
    mem = sh("free -g | awk '/Mem:/{print \"RAM 사용 \"$3\"G / \"$2\"G, 여유(available) \"$7\"G\"}'")
    gpu = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits")
    if gpu:
        u, m, t = [x.strip() for x in gpu.split(",")]
        gpu = "GPU 사용률 {}%, 메모리 {:.1f}/{:.1f}G".format(u, int(m) / 1024, int(t) / 1024)
    out.append("{}   {}".format(mem, gpu))
    killed = sh("journalctl -k --since '-2h' 2>/dev/null | grep -i 'killed process' | grep -c python")
    if killed and killed != "0":
        last = sh("journalctl -k --since '-2h' 2>/dev/null | grep -i 'killed process' | grep python | tail -1 | cut -c1-15")
        out.append("주의: 최근 2시간 RAM 부족(OOM)으로 python 강제 종료 {}건, 마지막 {}".format(killed, last))
    log = os.path.join(OUT, "pilots.log")
    if os.path.exists(log):
        lines = [l.rstrip() for l in open(log) if l.strip() and not l.startswith(("|", "#", "nohup", "Runs are"))]
        if lines:
            out.append("최근 기록 (outputs/pilots.log):")
            out += ["  " + l[:160] for l in lines[-4:]]
    return "\n".join(out)


def main():
    if "--watch" in sys.argv:
        i = sys.argv.index("--watch")
        sec = int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 60
        try:
            while True:
                s = status()
                print("\033[2J\033[H" + s + "\n\n({}초마다 갱신, Ctrl+C로 종료)".format(sec), flush=True)
                time.sleep(sec)
        except KeyboardInterrupt:
            pass
    else:
        print(status())


if __name__ == "__main__":
    main()
