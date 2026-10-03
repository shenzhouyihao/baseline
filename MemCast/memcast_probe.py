import os
import sys

print(sys.version)
print("OPENAI_API_KEY set:", bool(os.environ.get("OPENAI_API_KEY")))
print("OPENAI_BASE_URL:", os.environ.get("OPENAI_BASE_URL", ""))
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
except Exception as exc:
    print("torch error:", repr(exc))
for name in ("pandas", "openai", "dotenv", "sklearn", "tqdm"):
    try:
        module = __import__(name)
        print(name, "ok", getattr(module, "__version__", ""))
    except Exception as exc:
        print(name, "error", repr(exc))
