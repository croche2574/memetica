from md_helper import MoondreamHelper
from lancedb.embeddings import register, EmbeddingFunction
from pydantic import PrivateAttr
from PIL.Image import Image

@register('moondream2')
class MoondreamEmbeddings(EmbeddingFunction):
    name: str = 'moondream-2b-int8'
    device: str = 'cpu'
    _model = PrivateAttr()
    _ndims = PrivateAttr()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = MoondreamHelper()
        self._ndims = None
    
    def ndims(self):
        if self._ndims is None:
            self._ndims = {
                "image": self._model.gen_image_embed(Image.new(2,2)).shape[1],
                "string": self._model.gen_query_embed('foo').shape[2]
            }
        return self._ndims
    
    def compute_query_embeddings(self, query: str | Image):
        return self._model.gen_query_embed(query)

