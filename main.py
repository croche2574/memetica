import lancedb
from meme_schema import Meme
import os
from tqdm import tqdm

db_uri = './db/memetica-db'
model_uri= './models/moondream-2b-int8.mf'
weights_uri= './models/model.safetensors'

db = lancedb.connect(db_uri)

tbl = db.create_table('memes', schema=Meme, exist_ok=True, mode='overwrite')

def image_loader():
    current_dir = os.getcwd()
    images_folder = os.path.join(current_dir, "ingest")


    image_files = [filename for filename in os.listdir(images_folder)
        if filename.endswith((".png", ".jpg", ".jpeg"))]
    if len(image_files) > 1:
        for filename in tqdm(image_files, desc="loading images"):
            image_path = os.path.join(images_folder, filename)
            print(image_path)

            with open(image_path, 'rb') as f:
                yield Meme(image_bytes=f.read())
    elif len(image_files) == 1:
        image_path = os.path.join(images_folder, image_files[0])
        print(image_path)
        with open(image_path, 'rb') as f:
            yield [Meme(image_bytes=f.read())]
            

tbl.add(image_loader())

