import io

p = "reproduce.py"
s = io.open(p, encoding="utf-8").read()

# run_canonical_baselines and run_chemprop_v2_benchmarks accept 'all' directly
s = s.replace("""    "baselines": [
        ["benchmarks/run_canonical_baselines.py", "--dataset", name] for name in DATASETS
    ],""", """    "baselines": [
        ["benchmarks/run_canonical_baselines.py", "--dataset", "all"],
    ],""")
s = s.replace("""    "chemprop": [
        ["benchmarks/run_chemprop_v2_benchmarks.py", "--dataset", name] for name in DATASETS
    ],""", """    "chemprop": [
        ["benchmarks/run_chemprop_v2_benchmarks.py", "--dataset", "all"],
    ],""")
s = s.replace("""    "seed-variance": [
        ["benchmarks/run_seed_variance_decomposition.py"],
    ],""", """    "seed-variance": [
        ["benchmarks/run_seed_variance_decomposition.py", "--dataset", "bbbp", "--split-seed", "1"],
    ],""")

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched reproduce.py")
