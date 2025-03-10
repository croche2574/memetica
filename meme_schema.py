from pydantic import BaseModel
from lancedb.pydantic import Vector, LanceModel
import numpy as np
from typing import List

from lancedb.embeddings import get_registry

registry = get_registry()
clip = registry.get("open-clip").create()


class LiteralCaption(BaseModel):
    caption: str = clip.SourceField()
    vector: Vector(2048) = clip.VectorField()

class ConceptualCaption(BaseModel):
    caption: str = clip.SourceField()
    vector: Vector(2048) = clip.VectorField()

class Meme(LanceModel):
    image_bytes: bytes = clip.SourceField()
    image_vector: Vector(2048*720) = clip.VectorField()
    literal_capt: LiteralCaption
    conceptual_capt: ConceptualCaption
    content_tags: List[str]
    emotion_tags: List[str]
    meme_types: List[str]