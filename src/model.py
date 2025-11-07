from blocks.item import ItemEncoder, TimeAwareSessionEncoder, TitleDecoder
from flax import nnx
import jax
import jax.numpy as jnp


class Model(nnx.Module):
    """
    Full Hierarchical Time-Aware MoE model
    """

    def __init__(self, config: dict, vocab_sizes: dict, *, rngs: nnx.Rngs):
        self.config = config
        self.vocab_sizes = vocab_sizes

        self.item_encoder = ItemEncoder(config, vocab_sizes, rngs=rngs)
        self.session_encoder = TimeAwareSessionEncoder(config, rngs=rngs)
        self.title_decoder = TitleDecoder(config, vocab_sizes, rngs=rngs)

        self.rec_head = nnx.Linear(
            in_features=config["embed_dim"], out_features=vocab_sizes["id"], rngs=rngs
        )
        self.gen_head = nnx.Linear(
            in_features=config["embed_dim"],
            out_features=vocab_sizes["title"],
            rngs=rngs,
        )

    def __call__(self, inputs):
        session_item_inputs = inputs["session_items"]
        session_deltas = inputs["session_deltas"]
        target_title_input = inputs["target_title"]

        batch_size, session_len = session_item_inputs["title"].shape[0:2]

        # Item Encoder
        flat_inputs = jax.tree_util.tree_map(
            lambda x: x.reshape(batch_size * session_len, -1).squeeze(),
            session_item_inputs,
        )

        item_vectors_flat = self.item_encoder(flat_inputs)
        item_vectors = item_vectors_flat.reshape(
            (batch_size, session_len, self.config["embed_dim"])
        )

        # Session Encoder
        session_mask = (session_item_inputs["brand"] > 0).astype(jnp.int32)
        session_context = self.session_encoder(
            item_vectors, session_deltas, mask=jnp.expand_dims(session_mask, -2)
        )

        # Multi-Task Heads
        last_item_context = session_context[:, -1, :]
        rec_output = self.rec_head(last_item_context)

        decoder_output = self.title_decoder(
            target_title_input,
            session_context,
            context_mask=jnp.expand_dims(session_mask, -2),
        )

        gen_output = self.gen_head(decoder_output)

        return {"rec_output": rec_output, "gen_output": gen_output}
