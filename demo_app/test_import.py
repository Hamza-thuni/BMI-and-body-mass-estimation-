
import sys
import traceback
try:
    print('Starting import test', flush=True)
    import demo_app
    print('Import successful', flush=True)
except Exception as e:
    print('Error:', e, flush=True)
    traceback.print_exc()

