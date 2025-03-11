from pydantic import BaseModel
from lancedb.pydantic import Vector, LanceModel
import numpy as np
from typing import List
import md_embedder

from lancedb.embeddings import EmbeddingFunctionRegistry

registry = EmbeddingFunctionRegistry.get_instance()
md = registry.get("moondream2").create()


class LiteralCaption(BaseModel):
    caption: str = md.SourceField()
    vector: Vector(md.ndims()['text']) = md.VectorField()

class ConceptualCaption(BaseModel):
    caption: str = md.SourceField()
    vector: Vector(md.ndims()['text']) = md.VectorField()

class Meme(LanceModel):
    image_bytes: bytes = md.SourceField()
    image_vector: Vector(md.ndims()['image']) = md.VectorField()
    literal_capt: LiteralCaption
    conceptual_capt: ConceptualCaption
    content_tags: List[str]
    emotion_tags: List[str]
    meme_types: List[str]