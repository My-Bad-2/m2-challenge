from flax import nnx
import jax
import jax.numpy as jnp
from .encoder import StandardEncoderBlock, StandardDecoderBlock, TimeAwareEncoderBlock


class ItemEncoder(nnx.Module):
    """
    Encodes a single item's rich features
    """

    def __init__(self, config: dict, vocab_sizes: dict, *, rngs: nnx.Rngs):
        self.config = config
        self.vocab_sizes = vocab_sizes
        self.embed_dim = config["embed_dim"]

        # Text feature Embeddings
        self.title_embed = nnx.Embed(
            num_embeddings=vocab_sizes["title"], features=self.embed_dim, rngs=rngs
        )

        self.title_pos_embed = nnx.Embed(
            num_embeddings=config["max_title_len"], features=self.embed_dim, rngs=rngs
        )

        self.desc_embed = nnx.Embed(
            num_embeddings=vocab_sizes["desc"], features=self.embed_dim, rngs=rngs
        )

        self.desc_pos_embed = nnx.Embed(
            num_embeddings=config["max_desc_len"], features=self.embed_dim, rngs=rngs
        )

        # Categorical Feature Embeddings
        self.cat_features = [
            "brand",
            "category",
            "locale",
            "color",
            "size",
            "model",
            "material",
            "author",
        ]

        self.cat_embeds = {
            col: nnx.Embed(
                num_embeddings=vocab_sizes[col], features=self.embed_dim, rngs=rngs
            )
            for col in self.cat_features
        }

        # Numerical Feature Embedding (Price)
        self.price_bin_embed = nnx.Embed(
            num_embeddings=vocab_sizes["price_bin"], features=self.embed_dim, rngs=rngs
        )

        # Text Encoders
        self.title_encoder = StandardEncoderBlock(config, ffn_type="swiglu", rngs=rngs)
        self.desc_encoder = StandardEncoderBlock(config, ffn_type="swiglu", rngs=rngs)

        # Feature combiner
        self.feature_combiner = nnx.Linear(
            in_features=(len(self.cat_features) + 3)
            * self.embed_dim,  # 3 = title + desc + price
            out_features=self.embed_dim,
            rngs=rngs,
        )

        self.layer_norm = nnx.LayerNorm(
            num_features=self.embed_dim, epsilon=1e-6, rngs=rngs
        )

    def __call__(self, inputs):
        # Encode Title
        title_seq = inputs["title"]
        title_len = title_seq.shape[-1]
        title_mask = (title_seq > 0).astype(jnp.int32)
        title_pos = jnp.arange(0, title_len)
        title_vectors = self.title_embed(title_seq) + self.title_pos_embed(title_pos)
        title_context = self.title_encoder(
            title_vectors, mask=jnp.expand_dims(title_mask, -2)
        )
        pooled_title = title_context * jnp.expand_dims(title_mask, -1)
        pooled_title = pooled_title.sum(1) / jnp.maximum(
            title_mask.sum(-1, keepdims=True), 1
        )

        # Encode Description
        desc_seq = inputs["desc"]
        desc_len = desc_seq.shape[-1]
        desc_mask = (desc_seq > 0).astype(jnp.int32)
        desc_pos = jnp.arange(0, desc_len)
        desc_vectors = self.desc_embed(desc_seq) + self.desc_pos_embed(desc_pos)
        desc_context = self.desc_encoder(
            desc_vectors, mask=jnp.expand_dims(desc_mask, -2)
        )
        pooled_desc = desc_context * jnp.expand_dims(desc_mask, -1)
        pooled_desc = pooled_desc.sum(1) / jnp.maximum(
            desc_mask.sum(-1, keepdims=True), 1
        )

        # Embed all other features
        all_vecs = [pooled_title, pooled_desc]

        for col in self.cat_features:
            all_vecs.append(self.cat_embeds[col](inputs[col]))

        all_vecs.append(self.price_bin_embed(inputs["price_bin"]))

        # Combine
        combined_features = jnp.concatenate(all_vecs, axis=-1)

        projected_vector = self.feature_combiner(combined_features)
        projected_vector = jax.nn.relu(projected_vector)

        return self.layer_norm(projected_vector)


class TimeAwareSessionEncoder(nnx.Module):
    """
    Encodes a sequence of item vectors
    """

    def __init__(self, config: dict, *, rngs: nnx.Rngs):
        self.config = config
        self.pos_embed = nnx.Embed(
            num_embeddings=config["max_session_len"],
            features=config["embed_dim"],
            rngs=rngs,
        )

        self.encoder_blocks = [
            TimeAwareEncoderBlock(config, rngs=rngs)
            for _ in range(config["session_layers"])
        ]

        self.final_norm = nnx.LayerNorm(
            num_features=config["embed_dim"], epsilon=1e-6, rngs=rngs
        )

    def __call__(self, item_sequence_vectors, session_deltas, mask=None):
        session_len = item_sequence_vectors.shape[1]
        pos_ids = jnp.arange(0, session_len)
        session_vectors = item_sequence_vectors + self.pos_embed(pos_ids)

        x = session_vectors

        for block in self.encoder_blocks:
            x = block(x, session_deltas=session_deltas, mask=mask)

        return self.final_norm(x)


class TitleDecoder(nnx.Module):
    """
    Decodes a title
    """

    def __init__(self, config: dict, vocab_sizes: dict, *, rngs: nnx.Rngs):
        self.config = config
        self.token_embed = nnx.Embed(
            num_embeddings=vocab_sizes["title"], features=config["embed_dim"], rngs=rngs
        )
        self.pos_embed = nnx.Embed(
            num_embeddings=config["max_title_len"],
            features=config["embed_dim"],
            rngs=rngs,
        )

        self.decoder_blocks = [
            StandardDecoderBlock(config, ffn_type="moe", rngs=rngs)
            for _ in range(config["decoder_layers"])
        ]

        self.final_norm = nnx.LayerNorm(
            num_features=config["embed_dim"], epsilon=1e-6, rngs=rngs
        )

    def __call__(self, target_token_seq, encoder_context, context_mask=None):
        target_len = target_token_seq.shape[1]
        target_mask = (target_token_seq > 0).astype(jnp.int32)
        pos_ids = jnp.arange(0, target_len)
        target_vectors = self.token_embed(target_token_seq) + self.pos_embed(pos_ids)

        x = target_vectors

        for block in self.decoder_blocks:
            x = block(
                x,
                encoder_context,
                x_mask=jnp.expand_dims(target_mask, -2),
                context_mask=context_mask,
            )

        return self.final_norm(x)
