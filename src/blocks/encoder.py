import jax.numpy as jnp
from flax import nnx
from .gqa import SparseMoE, TimeAwareGQA, GQA, SwiGLU
from typing import Literal


class TimeAwareEncoderBlock(nnx.Module):
    """
    Transformer Encoder block that uses TimeAwareGQA
    """

    def __init__(self, config: dict, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(
            num_features=config["embed_dim"], epsilon=1e-6, rngs=rngs
        )
        self.attn = TimeAwareGQA(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_kv_groups=config["num_kv_groups"],
            rngs=rngs,
        )

        self.norm2 = nnx.LayerNorm(
            num_features=config["embed_dim"], epsilon=1e-6, rngs=rngs
        )
        self.moe = SparseMoE(
            num_experts=config["num_experts"],
            experts_per_token=config["experts_per_token"],
            embed_dim=config["embed_dim"],
            hidden_dim=config["embed_dim"] * 4,
            rngs=rngs,
        )

    def __call__(self, x, session_deltas, mask=None):
        x_norm = self.norm1(x)
        attn_output = self.attn(
            query=x_norm,
            value=x_norm,
            key=x_norm,
            session_deltas=session_deltas,
            mask=mask,
        )

        x = x + attn_output
        x_norm = self.norm2(x)
        ffn_output = self.moe(x_norm)
        x = x + ffn_output

        return x


class StandardEncoderBlock(nnx.Module):
    """
    A standard Transformer Encoder Block.
    """

    def __init__(
        self,
        config: dict,
        ffn_type: Literal["swiglu", "moe"] = "swiglu",
        *,
        rngs: nnx.Rngs
    ):
        self.norm1 = nnx.LayerNorm(
            num_features=config["embed_dim"], epsilon=1e-6, rngs=rngs
        )
        self.self_attn = GQA(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_kv_groups=config["num_kv_groups"],
            use_casual_mask=False,
            rngs=rngs,
        )

        self.norm2 = nnx.LayerNorm(
            num_features=config["embed_dim"], epsilon=1e-6, rngs=rngs
        )

        if ffn_type == "swiglu":
            self.ffn = SwiGLU(
                hidden_dim=config["embed_dim"] * 4,
                output_dim=config["embed_dim"],
                rngs=rngs,
            )
        elif ffn_type == "moe":
            self.ffn = SparseMoE(
                num_experts=config["num_experts"],
                experts_per_token=config["experts_per_token"],
                embed_dim=config["embed_dim"],
                hidden_dim=config["embed_dim"] * 4,
                rngs=rngs,
            )
        else:
            raise ValueError("ffn_type must be 'swiglu' or 'moe'")

    def __call__(self, x, mask=None):
        x_norm = self.norm1(x)
        attn_output = self.self_attn(query=x_norm, value=x_norm, key=x_norm, mask=mask)

        x = x + attn_output
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output
        return x


class StandardDecoderBlock(nnx.Module):
    """
    Standard Pre-LN Transformer Decoder block
    """

    def __init__(self, config: dict, ffn_type: Literal['swiglu', 'moe'] = 'moe', *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(
            num_features=config["embed_dim"], epsilon=1e-6, rngs=rngs
        )
        self.self_attn = GQA(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_kv_groups=config["num_kv_groups"],
            use_casual_mask=True,
            rngs=rngs,
        )

        self.norm2 = nnx.LayerNorm(config["embed_dim"], epsilon=1e-6, rngs=rngs)

        self.cross_attn = GQA(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_kv_groups=config["num_kv_groups"],
            use_casual_mask=False,
            rngs=rngs,
        )

        self.norm3 = nnx.LayerNorm(
            num_features=config["embed_dim"], epsilon=1e-6, rngs=rngs
        )
        
        if ffn_type == 'swiglu':
            self.ffn = SwiGLU(hidden_dim=config["embed_dim"] * 4, output_dim=config["embed_dim"], rngs=rngs)
        elif ffn_type == 'moe':
            self.ffn = SparseMoE(
                num_experts=config["num_experts"],
                experts_per_token=config["experts_per_token"],
                embed_dim=config["embed_dim"],
                hidden_dim=config["embed_dim"] * 4,
                rngs=rngs,
            )
        else:
            raise ValueError("ffn_type must be 'swiglu' or 'moe'")

    def __call__(self, x, encoder_context, x_mask=None, context_mask=None):
        x_norm = self.norm1(x)
        self_attn_output = self.self_attn(
            query=x_norm, value=x_norm, key=x_norm, mask=x_mask
        )

        x = x + self_attn_output

        x_norm = self.norm2(x)
        cross_attn_output = self.cross_attn(
            query=x_norm, value=encoder_context, key=encoder_context, mask=context_mask
        )

        x = x + cross_attn_output

        x_norm = self.norm3(x)
        ffn_output = self.ffn(x_norm)

        x = x + ffn_output
        return x
