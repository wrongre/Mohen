from fastapi import FastAPI
import uvicorn
app = FastAPI()

@app.get("/")
def root():
    return {"hello": "world"}

if __name__ == "__main__":
    print("starting test app")
    uvicorn.run(app, host="127.0.0.1", port=8001)
