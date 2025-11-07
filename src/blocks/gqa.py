import jax
import jax.numpy as jnp
from flax import nnx


class SwiGLU(nnx.Module):
    """
    SwiGLU Gated Feed-Forward Network
    """

    def __init__(self, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs):
        self.gate_proj = nnx.Linear(
            in_features=output_dim, out_features=hidden_dim, use_bias=False, rngs=rngs
        )

        self.up_proj = nnx.Linear(
            in_features=output_dim, out_features=hidden_dim, use_bias=False, rngs=rngs
        )

        self.down_proj = nnx.Linear(
            in_features=hidden_dim, out_features=output_dim, use_bias=False, rngs=rngs
        )

    def __call__(self, x):
        gate = self.gate_proj(x)
        gate = nnx.silu(gate)
        up = self.up_proj(x)
        fused_gate = gate * up
        return self.down_proj(fused_gate)


class SparseMoE(nnx.Module):
    """
    Sparse Mixture of Experts Layer with Top_k Gating
    """

    def __init__(
        self,
        num_experts: int,
        experts_per_token: int,
        embed_dim: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs
    ):
        self.num_experts = num_experts
        self.k = experts_per_token
        self.embed_dim = embed_dim

        self.router = nnx.Linear(embed_dim, num_experts, use_bias=False, rngs=rngs)

        self.experts = [
            SwiGLU(hidden_dim, embed_dim, rngs=rngs) for _ in range(num_experts)
        ]

    def __call__(self, x):
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(-1, self.embed_dim)

        router_logits = self.router(x_flat)
        top_k_logits, top_k_indices = jax.lax.top_k(router_logits, k=self.k)
        top_k_weights = nnx.softmax(top_k_logits, axis=-1)

        final_output = jnp.zeros_like(x_flat)

        for i in range(self.num_experts):
            expert_mask = jnp.any(top_k_indices == i, axis=-1)
            expert_indices = jnp.where(expert_mask)[0]

            if expert_indices.shape[0] == 0:
                continue

            expert_tokens = x_flat[expert_indices]
            expert_output = self.experts[i](expert_tokens)

            k_indices = jnp.where(top_k_indices == i)
            weights_for_expert = top_k_weights[k_indices]

            weighted_output = expert_output * jnp.expand_dims(
                weights_for_expert, axis=-1
            )
            final_output = final_output.at[expert_indices].add(weighted_output)

        return final_output.reshape((batch_size, seq_len, self.embed_dim))


class GQA(nnx.Module):
    """
    Standard Grouped-Query Attention
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_groups: int,
        use_casual_mask: bool,
        *,
        rngs: nnx.Rngs
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.use_casual_mask = use_casual_mask

        self.key_dim = embed_dim // num_heads

        if num_heads % num_kv_groups != 0:
            raise ValueError("num_heads must be divisible by num_kv_groups")

        self.num_kv_heads = self.num_kv_groups

        self.query_proj = nnx.Linear(
            in_features=embed_dim, out_features=embed_dim, use_bias=False, rngs=rngs
        )

        self.key_proj = nnx.Linear(
            in_features=embed_dim,
            out_features=self.key_dim * self.num_kv_heads,
            use_bias=False,
            rngs=rngs,
        )

        self.value_proj = nnx.Linear(
            in_features=embed_dim,
            out_features=self.key_dim * self.num_kv_heads,
            use_bias=False,
            rngs=rngs,
        )

        self.output_proj = nnx.Linear(
            in_features=embed_dim, out_features=embed_dim, use_bias=False, rngs=rngs
        )

        self.scale = 1.0 / (float(self.key_dim) ** 0.5)

    def __call__(self, query, key, value, mask=None):
        batch_size, seq_len_q, _ = query.shape
        seq_len_kv = key.shape[1]

        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(value)

        q = q.reshape(batch_size, seq_len_q, self.num_heads, self.key_dim).transpose(
            0, 2, 1, 3
        )
        k = k.reshape(
            batch_size, seq_len_kv, self.num_kv_heads, self.key_dim
        ).transpose(0, 2, 1, 3)
        v = v.reshape(
            batch_size, seq_len_kv, self.num_kv_heads, self.key_dim
        ).transpose(0, 2, 1, 3)

        n_rep = self.num_heads // self.num_kv_heads
        k_rep = jnp.repeat(k, n_rep, axis=1)
        v_rep = jnp.repeat(v, n_rep, axis=1)

        scores = jnp.matmul(q, k_rep.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            mask_expanded = jnp.expand_dims(jnp.expand_dims(mask, 1), 1)
            scores = scores + (1.0 - mask_expanded.astype(scores.dtype)) * -1e9

        if self.use_casual_mask:
            casual_mask = nnx.make_causal_mask(q, dtype=bool)
            scores = jnp.where(casual_mask, scores, -1e9)

        attention_weights = nnx.softmax(scores, axis=-1)  # type: ignore
        attention_output = jnp.matmul(attention_weights, v_rep)

        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len_q, self.embed_dim
        )

        return self.output_proj(attention_output)


class TimeAwareGQA(nnx.Module):
    """
    Time-Aware Grouped-Query Attention
    """

    def __init__(
        self, embed_dim: int, num_heads: int, num_kv_groups: int, *, rngs: nnx.Rngs
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.key_dim = embed_dim // num_heads

        if num_heads % num_kv_groups != 0:
            raise ValueError("num_heads must be divisible by num_kv_groups")

        self.num_kv_heads = self.num_kv_groups

        self.query_proj = nnx.Linear(
            in_features=embed_dim, out_features=embed_dim, use_bias=False, rngs=rngs
        )

        self.key_proj = nnx.Linear(
            in_features=embed_dim,
            out_features=self.key_dim * self.num_kv_heads,
            use_bias=False,
            rngs=rngs,
        )

        self.value_proj = nnx.Linear(
            in_features=embed_dim,
            out_features=self.key_dim * self.num_kv_heads,
            use_bias=False,
            rngs=rngs,
        )

        self.output_proj = nnx.Linear(
            in_features=embed_dim, out_features=embed_dim, use_bias=False, rngs=rngs
        )

        self.time_bias_weight = nnx.Param(nnx.initializers.zeros(rngs(), ()))

        self.scale = 1.0 / (float(self.key_dim) ** 0.5)

    def __call__(self, query, key, value, session_deltas, mask=None):
        batch_size, seq_len, _ = query.shape

        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(value)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.key_dim).transpose(
            0, 2, 1, 3
        )

        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.key_dim).transpose(
            0, 2, 1, 3
        )

        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.key_dim).transpose(
            0, 2, 1, 3
        )

        n_rep = self.num_heads // self.num_kv_heads
        k_rep = jnp.repeat(k, n_rep, axis=1)
        v_rep = jnp.repeat(v, n_rep, axis=1)

        scores = jnp.matmul(q, k_rep.transpose(0, 1, 3, 2)) * self.scale

        cumulative_time = jnp.cumsum(session_deltas, axis=1)
        t_i = jnp.expand_dims(cumulative_time, 2)
        t_j = jnp.expand_dims(cumulative_time, 1)
        pairwise_delta = jnp.abs(t_i - t_j)
        log_delta = jnp.log(1.0 + pairwise_delta)
        time_bias = self.time_bias_weight * log_delta

        scores = scores + jnp.expand_dims(time_bias, 1)

        if mask is not None:
            mask_expanded = jnp.expand_dims(jnp.expand_dims(mask, 1), 1)
            scores = scores + (1.0 - mask_expanded.astype(scores.dtype)) * -1e9

        attention_weights = nnx.softmax(scores, axis=-1)  # type: ignore
        attention_output = jnp.matmul(attention_weights, v_rep)

        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.embed_dim
        )

        return self.output_proj(attention_output)
