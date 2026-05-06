
import sys, os
print('Starting test', flush=True)
try:
    import torch
    print('Torch imported', flush=True)
except Exception as e:
    print('Torch err:', e, flush=True)

try:
    from v9_pipeline_wrapper import RealV9Pipeline
    print('Wrapper imported', flush=True)
    pipe = RealV9Pipeline()
    print('Pipeline created', flush=True)
except Exception as e:
    import traceback
    print('Pipeline err:', e, flush=True)
    traceback.print_exc()

