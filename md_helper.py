from PIL.Image import Image
import torch
from MoondreamTorch.moondream import MoondreamConfig, MoondreamModel, text_encoder
from MoondreamTorch.weights import load_weights_into_model
from typing import List, Tuple

class EncodedImage:
    pos: int
    caches: List[Tuple[torch.Tensor, torch.Tensor]]

class MoondreamHelper:
    def __init__(self, model_uri):
        self.config = MoondreamConfig()
        self.model = MoondreamModel(self.config)
        load_weights_into_model(model_uri, self.model)

    def gen_answer_embed(self, text: str):
        answer_tokens = self.model.tokenizer.encode(text).ids
        answer_emb = text_encoder(
            torch.tensor([[answer_tokens]], device=self.model.device),
            self.model.text,
        ).squeeze(0)
        return answer_emb

    def gen_query_embed(self, text: str):
        question_tokens = self.model.tokenizer.encode(text).ids
        question_emb = text_encoder(
            torch.tensor([[question_tokens]], device=self.model.device),
            self.model.text,
        ).squeeze(0)
        return question_emb

    def gen_image_embed(self, image: Image):
        return self.model._run_vision_encoder(image)

    def encode_image(self, image: Image | torch.Tensor): # Accepts the reconstructed vector from DB created by method above this.
        if image.type() == Image:
            return self.model.encode_image(image)
        elif image.type() == torch.Tensor:
            with torch.inference_mode():
                bos_emb = text_encoder(
                    torch.tensor([[self.config.tokenizer.bos_id]], device=self.model.device),
                    self.model.text,
                )
                inputs_embeds = torch.cat([bos_emb, image[None]], dim=1)
                mask = self.model.attn_mask[:, :, 0 : inputs_embeds.size(1), :]
                pos_ids = torch.arange(inputs_embeds.size(1), dtype=torch.long)
                self.model._prefill(inputs_embeds, mask, pos_ids)

                return EncodedImage(
                    pos=inputs_embeds.size(1),
                    caches=[
                        (
                            b.kv_cache.k_cache[:, :, : inputs_embeds.size(1), :].clone(),
                            b.kv_cache.v_cache[:, :, : inputs_embeds.size(1), :].clone(),
                        )
                        for b in self.model.text.blocks
                    ],
                )