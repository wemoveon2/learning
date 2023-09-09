from fastapi import FastAPI
from mangum import Mangum

from api.api_v2.api import router as api_router


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "This API provides services for analyzing text data."}

app.include_router(api_router, prefix = "/api/v2")

#handler = Mangum(app) 

if __file__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port = 8001, log_level = 'debug')
