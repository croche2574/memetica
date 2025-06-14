import lancedb
from meme_schema import Meme
from auth import AuthManager, User, Token
from datetime import timedelta
import os
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from db_helper import load_from_folder
import asyncio
from datetime import datetime
from typing import Annotated

ACCESS_TOKEN_EXPIRE_MINUTES = 30

db_uri = './db/memetica-db'
model_uri= './models/moondream-2b-int8.mf'
weights_uri= './models/model.safetensors'
images_folder = os.path.join(os.getcwd(), "ingest")

db = lancedb.connect(db_uri)
meme_tbl = db.create_table('memes', schema=Meme, exist_ok=True, mode="overwrite")
user_tbl = db.create_table('users', schema=User, exist_ok=True, mode="overwrite")
#tbl.create_fts_index(["literal_capt.caption", "conceptual_capt.caption"], replace=True)
#tbl.create_scalar_index("timestamp", replace=True)

app = FastAPI()
auth_manager = AuthManager(user_tbl)

@app.get("/memes/check_ingest")
async def check_folder():
    return {"response": load_from_folder(meme_tbl, images_folder)}

@app.get("/memes/")
async def query_memes(query: str = f"timestamp <= TIMESTAMP '{datetime.now().isoformat()}'", limit: int = 10):
    return [m.json for m in meme_tbl.search().where(query).limit(limit).to_pydantic(Meme)]

@app.get("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
) -> Token:
    user = auth_manager.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_manager.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")

@app.get("/users/me", response_model=User)
async def read_users_me(
    current_user: Annotated[User, Depends(auth_manager.get_current_active_user)]
):
    return [{"item_id": "Foo", "owner": current_user}]

#print(asyncio.run(check_folder()))
#print(asyncio.run(query_memes()))