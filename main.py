import lancedb
from meme_schema import Meme
from md_helper import MoondreamHelper
from PIL import Image
import time

db_uri = './db/memetica-db'
model_uri= './models/moondream-2b-int8.mf'
weights_uri= './models/model.safetensors'

db = lancedb.connect(db_uri)

tbl = db.create_table('memes', schema=Meme, exist_ok=True)


model = MoondreamHelper(weights_uri)

embedded_img = model.gen_query_embed(Image.new('RGB', (22,20)))
embeddend_text = model.gen_query_embed("foo")
print(embedded_img.shape)
print(embeddend_text.shape)
