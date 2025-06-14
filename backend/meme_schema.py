from pydantic import BaseModel
import pyarrow as pa
from lancedb.pydantic import Vector, LanceModel
from lancedb.embeddings import EmbeddingFunctionRegistry
from typing import List, Optional
from datetime import datetime
import md_embedder
from PIL.Image import Image, open
import io
import base64
import json

class BytesEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, bytes):
            return base64.b64encode(o).decode()
        return super().default(o)

registry = EmbeddingFunctionRegistry.get_instance()
md = registry.get("moondream2").create()

class LiteralCaption(BaseModel):
    caption: str = md.SourceField()
    vector: Vector(md.ndims()['text']) = md.VectorField(default=None)

class ConceptualCaption(BaseModel):
    caption: str = md.SourceField()
    vector: Vector(md.ndims()['text']) = md.VectorField(default=None)

class Meme(LanceModel):
    timestamp: datetime = datetime.now()
    image_bytes: bytes = md.SourceField()
    image_vector: Vector(md.ndims()['image']) = md.VectorField(default=None)
    literal_capt: LiteralCaption
    conceptual_capt: ConceptualCaption
    tags: Optional[List[str]] = None

    @property
    def json_response(self) -> str:
        return json.dumps({
            "img": self.image_bytes,
            "created": self.timestamp.isoformat(),
            "literal_caption": self.literal_capt.caption,
            "conceptual_caption": self.conceptual_capt.caption
        }, cls=BytesEncoder)

    @property
    def image(self) -> Image:
        return open(io.BytesIO(self.image_bytes.as_py()))