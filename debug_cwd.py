import os, traceback
path = r"C:\Users\wrong\Documents\Projects\100_Clonify-Ink\StrokeDiffusion\tmpcwd.txt"
with open(path,'w') as f:
    f.write('cwd:' + os.getcwd())
    f.write('\n')
    try:
        import web_UI.main
    except Exception:
        f.write(traceback.format_exc())
print('done')