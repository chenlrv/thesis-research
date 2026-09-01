import os, sys
print("python", sys.version)
print("CONDA_PREFIX", os.environ.get("CONDA_PREFIX"))
print("PATH has Library/bin:", any("Library\\bin" in p for p in os.environ.get("PATH","").split(os.pathsep)))
try:
    import torch
    print("torch OK", torch.__version__, "cuda", torch.cuda.is_available())
except Exception as e:
    print("torch FAIL:", repr(e))
try:
    import torch_geometric
    print("PyG OK", torch_geometric.__version__)
except Exception as e:
    print("PyG FAIL:", repr(e))
