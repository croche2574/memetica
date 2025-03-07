from pydantic import BaseModel
from lancedb.pydantic import Vector, LanceModel
import numpy as np
from typing import List

class LiteralCaption(BaseModel):
    caption: str
    vector: Vector(2048)

class ConceptualCaption(BaseModel):
    caption: str
    vector: Vector(2048)

class Meme(LanceModel):
    image_bytes: bytes
    image_vector: Vector(2048*720)
    literal_capt: LiteralCaption
    conceptual_capt: ConceptualCaption
    content_tags: List[str]
    emotion_tags: List[str]
    meme_types: List[str]