import fastapi, os, glob
out=[]
root=fastapi.__path__[0]
for path in glob.glob(os.path.join(root,'**','*.py'), recursive=True):
    try:
        text=open(path,'r',encoding='utf-8',errors='ignore').read()
    except:
        continue
    if 'python-multipart' in text:
        out.append(path)
with open('fastapi_matches.txt','w',encoding='utf-8') as f:
    for p in out:
        f.write(p + '\n')
print('done')