from fastapi import FastAPI, Depends, HTTPException, status
from .auth import get_current_user

app = FastAPI()

@app.get("/protected")
def protected_route(user: dict = Depends(get_current_user)):
    return {"message": f"Hello, {user.get('sub', 'anonymous')}!"}
