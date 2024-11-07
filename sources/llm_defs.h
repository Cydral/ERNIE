#ifndef LlmNet_H
#define LlmNet_H

/**
 * @file llm_defs.h
 * @brief Transformer model definitions using Dlib
 *
 * This file defines the architecture and components of a Transformer model,
 * including two main variants: one for training (t_*) and one for inference (i_*).
 * It implements multi-head attention mechanisms, feed-forward layers, and RMS
 * normalization.
 *
 * Current version (v1.1.4) features:
 * - Clear separation between training and inference components
 * - RMS normalization instead of Layer Norm
 * - Optimized residual connections
 * - Causal masking for self-attention
 */

//#include <dlib/dnn.h>
#include "dlib_ext.h"

namespace llm
{
    using namespace dlib;

    // Network architectural parameters
    const long vocab_size       = 40000;    // Vocabulary size
    const long number_of_blocks = 4;        // Number of stacked Transformer blocks
    const long number_of_heads  = 8;        // Number of parallel attention heads
    const long embedding_size   = 64;       // Embedding dimension (d_model)
    const long sequence_size    = 24;       // Maximum sequence length

    // Scale Weights Layer
    template <long d_k_>
    class scale_weights_ : public multiply_ {
    public:
        explicit scale_weights_() : multiply_(1.0f / std::sqrt(static_cast<float>(d_k_))) {}
    };

    template <long d_k, typename SUBNET>
    using scale_weights = add_layer<scale_weights_<d_k>, SUBNET>;

    // Positional Embeddings
    template <long nb_embeddings, long embedding_length, typename SUBNET>
    using positional_embeddings =
        multiply<positional_encodings<htan<embeddings<nb_embeddings, embedding_length, SUBNET>>>>;

    // Learned Positional Embeddings
    template <long nb_embeddings, long embedding_length, typename SUBNET>
    using learned_positional_embeddings =
        htan<add_prev9<linear<embedding_length, skip10<
        tag9<embeddings<nb_embeddings, embedding_length, tag10<SUBNET>>>>>>>;

    // Latent Space
    template <long embedding_length, typename SUBNET>
    using latent_space = fc<embedding_length, relu<bn_fc<fc<512, SUBNET>>>>;

    // Classification Head
    template <long num_logits, typename SUBNET>
    using classification_head = loss_multiclass_log<fc<num_logits, SUBNET>>;
    //using classification_head = loss_cross_entropy<linear<num_logits, SUBNET>>;

    namespace v1_1_4 {
        template <long seq_len, long d_model, typename SUBNET>
        using query = extract<0, 1, seq_len, d_model, SUBNET>;

        template <long seq_len, long d_model, typename SUBNET>
        using key = extract<seq_len * d_model, 1, seq_len, d_model, SUBNET>;

        template <long seq_len, long d_model, typename SUBNET>
        using value = extract<(seq_len * d_model) * 2, 1, seq_len, d_model, SUBNET>;

        /**
         * Multi-Head Attention Implementation
         *
         * Architecture breakdown from inner to outer layers:
         * 1. Input Processing:
         *    - RMS normalization of input
         *    - Linear projection to (Q,K,V) space (d_model * 3)
         *    - Split into separate Q, K, V tensors
         * 2. Attention Head Formation:
         *    - Split Q,K,V into nb_heads parallel attention heads
         *    - Transpose K for matrix multiplication compatibility
         * 3. Scaled Dot-Product Attention:
         *    - Q*K^T matrix multiplication (multm_prev4)
         *    - Scaling by 1/sqrt(d_k) to control gradient magnitude
         *    - Causal masking (tril_mask) for autoregressive property
         *    - Softmax normalization
         * 4. Value Integration:
         *    - Attention weights * V multiplication
         *    - Concatenation of all heads (hstack)
         * 5. Output Processing:
         *    - Final linear projection to d_model dimension
         *    - Residual connection (add_prev1)
         *
         * Training-specific features:
         *    - Dropout layers (10% on output, 5% on attention weights)
         *    - Inference version replaces dropout with multiply layers
         *
         * Template Parameters:
         * @param seq_len: Maximum sequence length
         * @param d_model: Model dimension (embedding size)
         * @param nb_heads: Number of attention heads
         * @param SUBNET: Input subnet type
         */
        template <long seq_len, long d_model, long nb_heads, typename SUBNET>
        using t_multihead_attention =
            add_prev1<
            dropout_rate<10, linear<d_model,
            hstack<
            multm_prev3<
            dropout_rate<5, softmaxm<tril_mask<
            scale_weights<d_model / nb_heads,
            multm_prev4<hsplit<nb_heads, query<seq_len, d_model, skip2<
            tag4<transpose<hsplit<nb_heads, key<seq_len, d_model, skip2<
            tag3<hsplit<nb_heads, value<seq_len, d_model,
            tag2<hsplit<3, linear_no_bias<d_model * 3, rms_norm<
            tag1<SUBNET>>>>>>>>>>>>>>>>>>>>>>>>>>;
        template <long seq_len, long d_model, long nb_heads, typename SUBNET>
        using i_multihead_attention =
            add_prev1<
            multiply<linear<d_model,
            hstack<
            multm_prev3<
            multiply<softmaxm<tril_mask<
            scale_weights<d_model / nb_heads,
            multm_prev4<hsplit<nb_heads, query<seq_len, d_model, skip2<
            tag4<transpose<hsplit<nb_heads, key<seq_len, d_model, skip2<
            tag3<hsplit<nb_heads, value<seq_len, d_model,
            tag2<hsplit<3, linear_no_bias<d_model * 3, rms_norm<
            tag1<SUBNET>>>>>>>>>>>>>>>>>>>>>>>>>>;

        /**
         * Feed-Forward Network Layer Implementation
         *
         * Standard Transformer FFN with enhanced processing:
         * 1. Input Processing:
         *    - RMS normalization of input
         *    - Input tagged for residual connection
         * 2. Two-Layer Network:
         *    - First linear layer expands to 4x dimension (d_model * 4)
         *    - GELU activation function for non-linearity
         *    - Second linear layer projects back to d_model dimension
         * 3. Regularization and Skip Connection:
         *    - 10% dropout during training (replaced by multiply in inference)
         *    - Residual connection (add_prev5) combining input with output
         * Architecture follows the original Transformer paper with modifications:
         * - RMS norm instead of layer normalization
         * - GELU instead of ReLU activation
         * - Optimized residual connections using tagged layers
         *
         * Template Parameters:
         * @param d_model: Model dimension (embedding size)
         * @param SUBNET: Input subnet type
         */
        template <long d_model, typename SUBNET>
        using t_feed_forward =
            add_prev5<            
            dropout_rate<10, linear<d_model, gelu<linear<d_model * 4, rms_norm<
            tag5<SUBNET>>>>>>>;
        template <long d_model, typename SUBNET>
        using i_feed_forward =
            add_prev5<
            multiply<linear<d_model, gelu<linear<d_model * 4, rms_norm<
            tag5<SUBNET>>>>>>>;

        template <long seq_len, long d_model, long nb_heads, typename SUBNET>
        using t_transformer =
            t_feed_forward<d_model,
            t_multihead_attention<seq_len, d_model, nb_heads, SUBNET>>;
        template <long seq_len, long d_model, long nb_heads, typename SUBNET>
        using i_transformer =
            i_feed_forward<d_model,
            i_multihead_attention<seq_len, d_model, nb_heads, SUBNET>>;
    }

    /**
     * Transformer Model Architecture
     *
     * Implements two network variants:
     * - trn_net_v1_1: Training network with dropout
     * - inf_net_v1_1: Optimized inference network
     *
     * Common structure:
     * 1. Input Layer: Accepts token sequences (integer matrices)
     * 2. Embeddings: Combines token embeddings and positional encoding
     * 3. Transformer Blocks (x4):
     *    - RMS Normalization
     *    - Multi-head attention with causal mask
     *    - Feed-forward Network (FFN)
     *      * 4x dimension expansion
     *      * GELU activation
     *    - Residual connections
     * 4. Classification: Next token prediction
     *
     * Specific features:
     * - Training version (t_*): Includes dropout layers
     * - Inference version (i_*): Uses multiply instead of dropout
     * - RMS normalization for stability
     * - Causal masking for autoregressive learning
     */
    template <typename SUBNET>
    using t_transformer_block = v1_1_4::t_transformer<sequence_size, embedding_size, number_of_heads, SUBNET>;
    template <typename SUBNET>
    using i_transformer_block = v1_1_4::i_transformer<sequence_size, embedding_size, number_of_heads, SUBNET>;

    using trn_net_v1_1 = classification_head<vocab_size,        
        repeat<number_of_blocks, t_transformer_block,
        positional_embeddings<vocab_size, embedding_size,
        input<matrix<int, 0, 1>>>>>;
    using inf_net_v1_1 = classification_head<vocab_size,
        repeat<number_of_blocks, i_transformer_block,
        positional_embeddings<vocab_size, embedding_size,
        input<matrix<int, 0, 1>>>>>;
}

#endif // LlmNet_H