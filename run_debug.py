import runpy, traceback
try:
    runpy.run_path('web_UI/main.py', run_name='__main__')
except Exception:
    traceback.print_exc()
