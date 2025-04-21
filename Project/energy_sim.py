"""
energy_sim.py
-------------
Toy data‑movement energy simulator
(Reproduces the trend of Fig. 10 in Sze et al. 2017)

Usage
-----
python energy_sim.py RS          # 指定資料流 (預設 RS)
python energy_sim.py --plot      # 同時畫出五種資料流長條圖
"""

import sys, argparse, numpy as np, matplotlib.pyplot as plt, os

# ---------------- 能量模型（pJ/bit ⇒ 用相對數字即可） ----------------
E = dict(ALU=1, RF=2, NoC=6, SRAM=6, DRAM=200)  # 參考用

PROFILE = dict(  # 各 Dataflow 在每層級的「讀寫次數倍率」
    TEMP=dict(ALU=1, RF=6,  NoC=10, SRAM=20, DRAM=160),
    WS  =dict(ALU=1, RF=4,  NoC=8,  SRAM=15, DRAM=100),
    OS  =dict(ALU=1, RF=4.5,NoC=9,  SRAM=15, DRAM=110),
    NLR =dict(ALU=1, RF=2,  NoC=12, SRAM=25, DRAM=140),
    RS  =dict(ALU=1, RF=3,  NoC=4,  SRAM=8,  DRAM=40),
)

def calc_total(df: str) -> float:
    """Return total normalized energy per MAC for chosen dataflow."""
    return sum(PROFILE[df].values())

def print_breakdown(df: str):
    print(f"=== Energy per MAC (normalized, ALU = 1) — {df} ===")
    for k, v in PROFILE[df].items():
        print(f"{k:5}: {v:>5.1f}")
    print(f"Total : {calc_total(df):.1f}\n")

def plot_all(outname="energy_plot.png"):
    dfs, totals = zip(*((d, calc_total(d)) for d in PROFILE))
    plt.figure(figsize=(6, 4))
    plt.bar(dfs, totals, color="slategray")
    plt.ylabel("Energy / MAC (normalized)")
    plt.title("Toy Energy vs. Dataflow")
    plt.tight_layout()
    plt.savefig(outname, dpi=150)
    plt.close()
    print(f"Saved  {outname}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataflow", nargs="?", default="RS",
                    help="TEMP | WS | OS | NLR | RS (default)")
    ap.add_argument("--plot", action="store_true",
                    help="Generate bar chart for all dataflows")
    args = ap.parse_args()

    df = args.dataflow.upper()
    if df not in PROFILE:
        sys.exit(f"[Error] Unknown dataflow '{df}'. Choose from {list(PROFILE)}")

    print_breakdown(df)

    if args.plot:
        plot_all()              # energy_plot.png 存在目前資料夾
