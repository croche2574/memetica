from md_helper import MoondreamHelper
from lancedb.embeddings import EmbeddingFunctionRegistry, EmbeddingFunction
from pydantic import PrivateAttr
from PIL.Image import Image
from typing import Union, List
import numpy as np
import pyarrow as pa

TEXT = Union[str, List[str], pa.Array, pa.ChunkedArray, np.ndarray]
IMAGES = Union[
    str, bytes, List[str], List[bytes], pa.Array, pa.ChunkedArray, np.ndarray
]
registry = EmbeddingFunctionRegistry.get_instance()
weights_uri= './models/model.safetensors'


@registry.register('moondream2')
class MoondreamEmbeddings(EmbeddingFunction):
    name: str = 'moondream-2b-int8'
    device: str = 'cpu'
    _model = PrivateAttr()
    _ndims = PrivateAttr()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = MoondreamHelper(weights_uri)
        self._ndims = None
    
    def ndims(self):
        if self._ndims is None:
            self._ndims = {
                "image": 1492992,   #self._model.gen_image_embed(Image.new(2,2)).shape[1],
                "text": 2048        #self._model.gen_query_embed('foo').shape[2]
            }
        return self._ndims
    
    def compute_query_embeddings(self, query: str | Image):
        return self._model.gen_query_embed(query)
    
    def generate_text_embeddings(self, text: str):
        return self._model.gen_answer_embed(text)
    
    def compute_source_embeddings(self, inputs: TEXT | IMAGES) -> List[np.array]:
        embeddings = []
        for i in inputs:
            if isinstance(i, str):
                embeddings.append(self._model.gen_answer_embed(i))
            else:
                embeddings.append(self._model.gen_image_embed(i))
        return embeddings
