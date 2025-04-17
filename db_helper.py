from tqdm import tqdm
from meme_schema import Meme
import os
from lancedb.table import Table

def image_loader(image_files, folder_path):
    if len(image_files) > 1:
        for filename in tqdm(image_files, desc="loading images"):
            image_path = os.path.join(folder_path, filename)
            print(image_path)

            with open(image_path, 'rb') as f:
                meme = Meme(image_bytes=f.read())
                f.close()
                try:
                    os.remove(image_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (image_path, e))
                yield meme

    elif len(image_files) == 1:
        image_path = os.path.join(folder_path, image_files[0])
        print(image_path)
        with open(image_path, 'rb') as f:
            meme = [Meme(image_bytes=f.read())]
            f.close()
            try:
                os.remove(image_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (image_path, e))
            
            yield meme

def load_from_folder(tbl: Table, path: str) -> str:
    image_files = [filename for filename in os.listdir(path) if filename.endswith((".png", ".jpg", ".jpeg"))]

    if len(image_files) > 0:
        tbl.add(image_loader(image_files, path))
        return ("Loaded " + str(len(image_files)) + " memes from " + path)
    else:
        return "No new memes found."