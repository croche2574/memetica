from pydantic import BaseModel
from lancedb.pydantic import Vector, LanceModel
from typing import List, Optional
from datetime import datetime
import md_embedder
from PIL.Image import Image, open
import io, base64, json

class BytesEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, bytes):
            return base64.b64encode(o).decode()
        return super().default(o)

from lancedb.embeddings import EmbeddingFunctionRegistry

registry = EmbeddingFunctionRegistry.get_instance()
md = registry.get("moondream2").create()


class LiteralCaption(BaseModel):
    caption: str = md.SourceField()
    vector: Vector(md.ndims()['text']) = md.VectorField(default=None)

class ConceptualCaption(BaseModel):
    caption: str = md.SourceField()
    vector: Vector(md.ndims()['text']) = md.VectorField(default=None)

class Metadata(BaseModel):
    timestamp: datetime

class Meme(LanceModel):
    metadata: Metadata = Metadata(timestamp=datetime.now())
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
        }, cls=BytesEncoder)

    @property
    def image(self) -> Image:
        return open(io.BytesIO(self.image_bytes.as_py()))