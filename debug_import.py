import traceback
try:
    import web_UI.main
except Exception as e:
    with open('errorlog.txt','w', encoding='utf-8') as f:
        f.write(traceback.format_exc())
    print('error logged')
