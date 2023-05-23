from fastapi import FastAPI, Request
import numpy as np
from pydantic.main import BaseModel
import uvicorn
import joblib
from fastapi.templating import Jinja2Templates


templates = Jinja2Templates(directory="templates")

app = FastAPI(title = "My App")

#End point oluşturma
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})


@app.get("/predict/")
async def predict(request: Request,
                  l1:float, l2:float, l3:float, l4:float ):
    my_model = joblib.load("my_model")
    X = [l1,l2,l3,l4]
    X =np.array(X).reshape(-1,4)
    predicted = my_model.predict(X)

    return templates.TemplateResponse("prediction.html", {"request": request,
                                                          "predicted": predicted})

    

if __name__ == "__main__":
    uvicorn.run(app, host= "0.0.0.0") 