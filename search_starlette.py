import starlette, glob, os
root=starlette.__path__[0]
for path in glob.glob(os.path.join(root,'**','*.py'), recursive=True):
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        text=f.read()
    if 'multipart' in text.lower():
        print(path)
