import jax.numpy as jnp
import flax.linen as nnx
from .gqa import SparseMoE, TimeAwareGQA, GQA


class TimeAwareEncoderBlock(nnx.Module):
    """
    Transformer Encoder block that uses TimeAwareGQA
    """

    config: dict

    @nnx.compact
    def __call__(self, x, session_deltas, mask=None):
        embed_dim = self.config["embed_dim"]
        num_heads = self.config["num_heads"]
        num_kv_groups = self.config["num_kv_groups"]
        num_experts = self.config["num_experts"]
        experts_per_token = self.config["experts_per_token"]
        ffn_hidden_dim = embed_dim * 4

        norm1 = nnx.LayerNorm(epsilon=1e-6)
        attn = TimeAwareGQA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_groups=num_kv_groups,
            name="time_aware_gqa",
        )

        norm2 = nnx.LayerNorm(epsilon=1e-6)
        moe = SparseMoE(
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            embed_dim=embed_dim,
            hidden_dim=ffn_hidden_dim,
            name="sparse_moe_ffn",
        )

        x_norm = norm1(x)
        attn_output = attn(
            query=x_norm,
            value=x_norm,
            key=x_norm,
            session_deltas=session_deltas,
            mask=mask,
        )

        x = x + attn_output
        x_norm = norm2(x)
        ffn_output = moe(x_norm)
        x = x + ffn_output

        return x


class StandardDecoderBlock(nnx.Module):
    """
    Standard Pre-LN Transformer Decoder block
    """

    config: dict

    @nnx.compact
    def __call__(self, x, encoder_context, x_mask=None, context_mask=None):
        embed_dim = self.config["embed_dim"]
        num_heads = self.config["num_heads"]
        num_kv_groups = self.config["num_kv_groups"]
        num_experts = self.config["num_experts"]
        experts_per_token = self.config["experts_per_token"]
        ffn_hidden_dim = embed_dim * 4

        norm1 = nnx.LayerNorm(epsilon=1e-6)
        self_attn = GQA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_groups=num_kv_groups,
            use_casual_mask=True,
            name="self_attn_gqa",
        )

        norm2 = nnx.LayerNorm(epsilon=1e-6)
        cross_attn = GQA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_groups=num_kv_groups,
            use_casual_mask=False,
            name="cross_attn_gqa",
        )

        norm3 = nnx.LayerNorm(epsilon=1e-6)
        moe = SparseMoE(
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            embed_dim=embed_dim,
            hidden_dim=ffn_hidden_dim,
            name="sparse_moe_ffn",
        )

        x_norm = norm1(x)
        self_attn_output = self_attn(
            query=x_norm, value=x_norm, key=x_norm, mask=x_mask
        )

        x = x + self_attn_output

        x_norm = norm2(x)
        cross_attn_output = cross_attn(
            query=x_norm, value=encoder_context, key=encoder_context, mask=context_mask
        )

        x = x + cross_attn_output

        x_norm = norm3(x)
        ffn_output = moe(x_norm)

        x = x + ffn_output
        return x
