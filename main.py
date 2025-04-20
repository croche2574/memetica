import lancedb
from meme_schema import Meme
import os
import pandas as pd
from fastapi import FastAPI
from db_helper import load_from_folder
import asyncio
from datetime import datetime

db_uri = './db/memetica-db'
model_uri= './models/moondream-2b-int8.mf'
weights_uri= './models/model.safetensors'
images_folder = os.path.join(os.getcwd(), "ingest")

db = lancedb.connect(db_uri)
tbl = db.create_table('memes', schema=Meme, exist_ok=True)
#tbl.create_fts_index(["literal_capt.caption", "conceptual_capt.caption"], replace=True)
#tbl.create_scalar_index("timestamp", replace=True)

app = FastAPI()

@app.get("/memes/check_ingest")
async def check_folder():
    return {"response": load_from_folder(tbl, images_folder)}

print(asyncio.run(check_folder()))

time = datetime.now()

#print(tbl.to_pandas()["metadata"][0]["timestamp"])
memes = tbl.search().where(f"timestamp <= TIMESTAMP '{time.isoformat()}'").to_pydantic(Meme)
