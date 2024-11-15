#ifndef LlmNet_H
#define LlmNet_H

/**
 * @file llm_defs.h
 * @brief Optimized Transformer neural architecture for language processing
 *
 * Implements a Transformer architecture with multi-head attention and RMS
 * normalization, designed for efficient learning and inference. The architecture
 * leverages cognitive principles of parallel information processing and
 * selective attention.
 *
 * Key features v1.1.4:
 * - RMS normalization for enhanced stability
 * - Optimized residual connections
 * - Causal masking for autoregressive attention
 * - Dynamic weight adaptation through learning
 */

//#include <dlib/dnn.h>
#include "dlib_ext.h"

namespace llm
{
    using namespace dlib;

    // Network architectural parameters
    const long vocab_size       = 12000;    // Vocabulary size
    const long number_of_blocks = 4;        // Number of stacked Transformer blocks
    const long number_of_heads  = 8;        // Number of parallel attention heads
    const long embedding_size   = 256;      // Embedding dimension (d_model)
    const long sequence_size    = 32;       // Maximum sequence length

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
    using positional_embeddings = positional_encodings<embeddings<nb_embeddings, embedding_length, SUBNET>>;

    // Learned Positional Embeddings
    template <long nb_embeddings, long embedding_length, typename SUBNET>
    using learned_positional_embeddings =
        htan<add_prev9<linear<embedding_length, skip10<
        tag9<embeddings<nb_embeddings, embedding_length, tag10<SUBNET>>>>>>>;

    // Classification Head
    template <long num_logits, long embedding_length, typename SUBNET>
    using classification_head = loss_cross_entropy<linear<num_logits, rms_norm<SUBNET>>>;

    namespace v1_1_6 {
        template <long seq_len, long d_model, typename SUBNET>
        using query = extract<0, 1, seq_len, d_model, SUBNET>;

        template <long seq_len, long d_model, typename SUBNET>
        using key = extract<seq_len * d_model, 1, seq_len, d_model, SUBNET>;

        template <long seq_len, long d_model, typename SUBNET>
        using value = extract<(seq_len * d_model) * 2, 1, seq_len, d_model, SUBNET>;

        /**
         * Multi-Head Attention Implementation
         *
         * Models selective attention and parallel processing:
         * 1. Initial Processing:
         *    - RMS normalization to stabilize neural activation
         *    - Linear projection to (Q,K,V) space for feature extraction
         *    - Split into distinct Q, K, V tensors
         * 2. Attention Head Formation:
         *    - Split into nb_heads parallel processors
         *    - K transposition for matrix compatibility
         *    - Models brain's distributed processing
         * 3. Scaled Dot-Product Attention:
         *    - Similarity computation via Q*K^T
         *    - 1/sqrt(d_k) scaling for gradient stability
         *    - Causal masking for autoregressive learning
         *    - Softmax normalization of attention weights
         * 4. Value Integration:
         *    - Attention * values weighting
         *    - Head fusion through concatenation
         *    - Multi-channel information aggregation
         * 5. Final Processing:
         *    - Linear projection to model dimension
         *    - Residual connection for gradient flow
         * Training Specifics:
         *    - 10% output dropout, 5% attention weights dropout
         *    - Inference version: dropout -> "temperature value"
         *
         * Template Parameters:
         * @param seq_len: Maximum sequence length
         * @param d_model: Model dimension
         * @param nb_heads: Number of attention heads
         * @param SUBNET: Input subnet type
         */
        template <long seq_len, long d_model, long nb_heads, typename SUBNET>
        using multihead_attention =
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

        /**
         * Feed-Forward Network Implementation
         *
         * Enhanced standard feed-forward network:
         * 1. Preprocessing:
         *    - RMS normalization of input
         *    - Input tagged for residual connection
         * 2. Two-Layer Network:
         *    - First layer 4x dimension expansion
         *    - GELU non-linear activation
         *    - Projection back to model dimension
         * 3. Regularization and Residuals:
         *    - 8% dropout during training
         *    - add_prev5 residual connection
         * Architectural modifications:
         * - RMS norm for stability
         * - GELU for smooth non-linearity
         * - Optimized residual connections
         *
         * @param d_model: Model dimension
         * @param SUBNET: Input subnet type
         */
        template <long d_model, typename SUBNET>
        using feed_forward_fc =
            add_prev5<
            scale5<con<1, 1, 1, 1, 1,
            dropout_rate<8, fc<d_model, gelu<bn_fc<fc<d_model * 4, rms_norm<
            tag5<SUBNET>>>>>>>>>>;
        template <long d_model, typename SUBNET>
        using feed_forward =
            add_prev5<            
            dropout_rate<8, linear<d_model, gelu<linear<d_model * 4, rms_norm<
            tag5<SUBNET>>>>>>>;

        template <long seq_len, long d_model, long nb_heads, typename SUBNET>
        using transformer =
            feed_forward<d_model,
            multihead_attention<seq_len, d_model, nb_heads, SUBNET>>;
    }

    /**
     * Global Transformer Architecture
     *
     * Network variants:
     * - net_v1_1: Training&inferance network with/without dropout
     *
     * Common structure:
     * 1. Input Layer: Token sequences (integer matrices)
     * 2. Embeddings: Combines token and positional
     * 3. Transformer Blocks (x4):
     *    - Stabilizing RMS normalization
     *    - Causal multi-head attention
     *    - Feed-forward network:
     *      * 4x dimension expansion
     *      * GELU activation
     *    - Residual connections
     * 4. Classification: Next token prediction
     *
     * Specifics:
     * - Training version: Includes dropout
     * - Inference version: Uses a specific "temperature"
     * - RMS normalization for stability
     * - Causal masking for autoregression
     */
    template <typename SUBNET>
    using transformer_block = v1_1_6::transformer<sequence_size, embedding_size, number_of_heads, SUBNET>;

    using net_v1_1 = classification_head<vocab_size, embedding_size,
        repeat<number_of_blocks, transformer_block,
        positional_embeddings<vocab_size, embedding_size,
        input<matrix<int, 0, 1>>>>>;
}

#endif // LlmNet_H