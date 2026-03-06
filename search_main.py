keywords=["UploadFile","File(","Form"]
with open('web_UI/main.py','r',encoding='utf-8',errors='ignore') as f:
    lines=f.readlines()
out=[]
for i,l in enumerate(lines):
    for kw in keywords:
        if kw in l:
            out.append(f"{i+1}: {l.strip()}")
with open('main_matches.txt','w',encoding='utf-8') as f:
    f.write("\n".join(out))
print('written')
