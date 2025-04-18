from pydantic import BaseModel
from lancedb.pydantic import Vector, LanceModel
from typing import List, Optional
import md_embedder
import json
from PIL.Image import Image, open
import io

from lancedb.embeddings import EmbeddingFunctionRegistry

registry = EmbeddingFunctionRegistry.get_instance()
md = registry.get("moondream2").create()


class LiteralCaption(BaseModel):
    caption: str = md.SourceField()
    vector: Vector(md.ndims()['text']) = md.VectorField(default=None)

class ConceptualCaption(BaseModel):
    caption: str = md.SourceField()
    vector: Vector(md.ndims()['text']) = md.VectorField(default=None)

class Meme(LanceModel):
    image_bytes: bytes = md.SourceField()
    image_vector: Vector(md.ndims()['image']) = md.VectorField(default=None)
    literal_capt: Optional[LiteralCaption] = None
    conceptual_capt: Optional[ConceptualCaption] = None
    tags: Optional[List[str]] = None

    @property
    def json(self) -> str:
        return json.dumps({
            "img": self.image_bytes,
            "literal_caption": self.literal_capt.caption,
            "conceptual_caption": self.conceptual_capt.caption
        })

    @property
    def image(self) -> Image:
        return open(io.BytesIO(self.image_bytes.as_py()))