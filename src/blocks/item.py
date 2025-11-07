from flax import nnx
import jax

class ItemEncoder(nnx.Module):
    config: dict
    vocab_sizes: dict
    
    @nnx.compact
    def __call__(self, inputs):
        embed_dim = self.config['embed_dim']
        
        title_embed = nnx.Embed(self.vocab_sizes['title'], embed_dim)
        title_pos_embed = nnx.Embed(self.config['max_title_len'], embed_dim)
        title_encoder = nnx.TransformerEncoder()