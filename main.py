import lancedb
from meme_schema import Meme
from PIL import Image
import time

db_uri = './db/memetica-db'
model_uri= './models/moondream-2b-int8.mf'
weights_uri= './models/model.safetensors'

db = lancedb.connect(db_uri)

tbl = db.create_table('memes', schema=Meme, exist_ok=True)