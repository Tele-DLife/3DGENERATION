import torch

try:
    from transformers import BertModel
except Exception:
    BertModel = None

if BertModel is not None and not hasattr(BertModel, "get_head_mask"):
    def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is None:
            return [None] * num_hidden_layers
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        else:
            raise ValueError(f"head_mask.dim != 1/2, got {head_mask.dim()}")
        head_mask = head_mask.to(dtype=self.dtype)
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
        return head_mask

    BertModel.get_head_mask = _get_head_mask
