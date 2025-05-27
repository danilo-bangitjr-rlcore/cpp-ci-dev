from pathlib import Path

from ml_instrumentation.reader import load_all_results

path = Path('results/test/results.db')
df = load_all_results(path)
print(df)
