import jax
import jax.numpy as jnp
import flax.linen as nnx

class SwiGLU(nnx.Module):
    hidden_dim: int
    output_dim: int
    
    @nnx.compact
    def __call__(self, x):
        gate_projection = nnx.Dense(
            self.hidden_dim,
            use_bias=False,
            name="gate_projection"
        )
        up_projection = nnx.Dense(
            self.hidden_dim,
            use_bias=False,
            name="up_projection"
        )
        down_projection = nnx.Dense(
            self.output_dim,
            use_bias=False,
            name="down_projection"
        )
        
        gate = gate_projection(x)
        gate = nnx.silu(gate)
        up = up_projection(x)
        fused_gate = gate * up
        return down_projection(fused_gate)
    
class SparseMoE(nnx.Module):
    """
    Sparse Mixture of Experts Layer with Top_k Gating
    """
    num_experts: int
    experts_per_token: int
    embed_dim: int
    hidden_dim: int
    
    @nnx.compact
    def __call__(self, x):
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(-1, self.embed_dim)
        num_tokens = x_flat.shape[0]
        
        router = nnx.Dense(
            self.num_experts,
            use_bias=False,
            name="router"
        )
        
        experts = [
            SwiGLU(self.hidden_dim, self.embed_dim, name=f"expert_{i}")
            for i in range(self.num_experts)
        ]
        
        router_logits = router(x_flat)
        top_k_logits, top_k_indices = jax.lax.top_k(router_logits, k=self.experts_per_token)
        top_k_weights = nnx.softmax(top_k_logits, axis=-1)
        
        final_output = jnp.zeros_like(x_flat)
        
        for i in range(self.num_experts):
            expert_mask = jnp.any(top_k_indices == i, axis=-1)
            expert_indices = jnp.where(expert_mask)[0]
            
            if expert_indices.shape[0] == 0:
                continue
            
            expert_tokens = x_flat[expert_indices]
            expert_output = experts[i](expert_tokens)
            
            k_indices = jnp.where(top_k_indices == i)
            weights_for_expert = top_k_weights[k_indices]
            
            weighted_output = expert_output * jnp.expand_dims(weights_for_expert, axis=-1)
            final_output = final_output.at[expert_indices].add(weighted_output)
            
        return final_output.reshape((batch_size, seq_len, self.embed_dim))
    
