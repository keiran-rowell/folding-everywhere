// src/config.rs

#[derive(Clone, Copy, Debug)]
pub struct EsmConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
}

impl EsmConfig {
    pub fn esm2_650m() -> Self {
        Self {
            num_layers: 33,
            hidden_size: 1280,
            num_heads: 20,
            intermediate_size: 5120,
        }
    }

    pub fn esm2_3b() -> Self {
        Self {
            num_layers: 36,
            hidden_size: 2560,
            num_heads: 40,
            intermediate_size: 10240,
        }
    }

    /// Auto-detect model architecture from safetensors index or embedding dimensions
    pub fn from_weights(weights: &crate::weights::Weights) -> Self {
        // e.g. check word embeddings shape [vocab_size, hidden_dim]
        if let Some(entry) = weights.index.get("esm.embeddings.word_embeddings.weight") {
            let hidden_dim = entry.shape[1];
            match hidden_dim {
                1280 => Self::esm2_650m(),
                2560 => Self::esm2_3b(),
                other => panic!("Unsupported hidden dimension: {other}"),
            }
        } else {
            // Default fallback
            Self::esm2_650m()
        }
    }
}
