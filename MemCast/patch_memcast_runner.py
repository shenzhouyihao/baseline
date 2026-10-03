from pathlib import Path
p = Path('/home/lzm/work/MemCast/memcast_experiment_runner.py')
s = p.read_text()
s = s.replace('"mse": f"{avg_mse:.10f}",\n                    "mae": f"{avg_mae:.10f}",', '"mse": f"{avg_mse:.6f}",\n                    "mae": f"{avg_mae:.6f}",')
p.write_text(s)
