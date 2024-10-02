#ifndef LlmNet_H
#define LlmNet_H

/**
 * @file llm_defs.h
 * @brief Definitions for Large Language Models (LLM) usable with Dlib
 *
 * This file contains the basic definitions, constants, and parameters
 * necessary to implement Transformer-type models for text generation
 * using the Dlib library. It defines the fundamental structure of LLM
 * networks, including vocabulary size, sequence length, number of
 * attention heads and Transformer blocks, as well as embedding size.
 *
 * @note These parameters can be adjusted based on specific model requirements
 * and available computational resources.
 */

//#include <dlib/dnn.h>
#include "dlib_ext.h"

namespace llm
{
    using namespace dlib;

    // Global parameters for the Transformer network
    const long vocab_size = 8000;                                            // Size of the vocabulary
    const long number_of_blocks = 2;                                         // Number of transformer blocks
    const long number_of_heads = 16;                                         // Number of attention heads
    const long embedding_size = (480 / number_of_heads) * number_of_heads;   // Size of the embedding
    const long sequence_size = (embedding_size / number_of_heads);           // Length of the sequence

    // Scale Weights Layer
    // This layer scales the attention weights by a factor of 1/sqrt(d_k),
    // where d_k is the dimension of each attention head
    template <unsigned long number_of_heads_, unsigned long embedding_dim_>
    class scale_weights_ : public multiply_ {
        static_assert(number_of_heads_ > 0, "The number of heads must be > 0");
        static_assert(embedding_dim_ > 0, "The embeddind size must be > 0");

    public:
        explicit scale_weights_() : multiply_(1.0f / std::sqrt(static_cast<float>(embedding_dim_ / number_of_heads_))) {}
    };

    // Helper alias template for easier use of the scale_weights layer
    template <unsigned long num_heads, unsigned long embedding_length, typename SUBNET>
    using scale_weights = add_layer<scale_weights_<num_heads, embedding_length>, SUBNET>;

    // Positional Embeddings
    // This combines token embeddings with positional encodings
    template <unsigned long nb_embeddings, unsigned long embedding_length, typename SUBNET>
    using positional_embeddings =
        positional_encodings<  // Add positional information to the embeddings
        htan<  // Apply hyperbolic tangent activation
        embeddings<nb_embeddings, embedding_length, SUBNET>>>;  // Create token embeddings

    // Classification head
    // This adds a final fully connected layer for classification
    template <long num_logits, typename SUBNET>
    using classification_head = loss_multiclass_log<fc<num_logits, rms_norm<SUBNET>>>;

    namespace v1_0_6 {
        // Basic layers for "Query", "Key", and "Value"
        // These are 1x1 convolutions that project the input into different spaces
        template <long nb_heads, typename SUBNET>
        using query = con<nb_heads, 1, 1, 1, nb_heads, SUBNET>;

        template <long nb_heads, typename SUBNET>
        using key = con<nb_heads, 1, 1, 1, nb_heads, SUBNET>;

        template <long nb_heads, typename SUBNET>
        using value = con<nb_heads, 1, 1, 1, nb_heads, SUBNET>;

        // Core masked multihead attention block
        // This implements the core of the transformer's attention mechanism
        template <long embedding_dim, long nb_heads, typename SUBNET>
        using multihead_attention =
            add_prev3<  // Add the output to the input (residual connection)
            cont<1, 1, nb_heads, 1, nb_heads,  // Combine heads
            multm_prev1<  // Matrix multiplication with value
            dropout_rate<10,  // Apply dropout
            softmaxm<  // Apply softmax to attention scores
            tril_mask<  // Apply triangular mask for causal attention
            scale_weights<nb_heads, embedding_dim,  // Scale dot products
            multm_prev2<  // Matrix multiplication of query and key
            query<nb_heads, skip3<  // Query transformation
            tag2<transpose<  // Transpose key for dot product
            key<nb_heads, skip3<  // Key transformation
            tag1<value<nb_heads,  // Value transformation
            tag3<SUBNET>>>>>>>>>>>>>>>>>;

        // Feedforward blocks
        // This implements the position-wise feedforward network in the transformer
        template <long sequence_dim, long embedding_dim, typename SUBNET>
        using feed_forward_fc =
            add_prev5<
            scale5<con<1, 1, 1, 1, 1,
            fc<embedding_size,
            dropout_rate<10, gelu<bn_fc<fc<embedding_size / 4,
            avg_pool_everything<tag5<SUBNET>>>>>>>>>>;

        template <long embedding_dim, typename SUBNET>
        using feed_forward =
            add_prev5<  // Add the output to the input (residual connection)
            linear<embedding_dim,  // Project back to embedding size
            dropout_rate<10,  // Apply dropout
            gelu<linear<embedding_dim * 4,  // Expand and apply GELU activation
            tag5<SUBNET>>>>>>;

        // Transformer block
        // This defines a complete transformer block, combining attention and feedforward layers
        template <typename SUBNET>
        using transformer_block =
            feed_forward<embedding_size,  // Apply the feedforward network
            multihead_attention<embedding_size, number_of_heads,  // Apply multihead attention
            rms_norm<SUBNET>>>;  // Apply RMS normalization before the block
    }

    namespace v1_1_4 {
        template <long num_filters_out, typename SUBNET>
        using query = linear_no_bias<num_filters_out, SUBNET>;

        template <long num_filters_out, typename SUBNET>
        using key = linear_no_bias<num_filters_out, SUBNET>;

        template <long num_filters_out, typename SUBNET>
        using value = linear_no_bias<num_filters_out, SUBNET>;

        template <long embedding_dim, long nb_heads, typename SUBNET>
        using multihead_attention =
            add_prev3<
            linear<embedding_dim,
            hstack<
            multm_prev1<
            softmaxm<tril_mask<
            scale_weights<nb_heads, embedding_dim,
            multm_prev2<hsplit<nb_heads, query<embedding_dim, skip3<
            tag2<transpose<hsplit<nb_heads, key<embedding_dim, skip3<
            tag1<hsplit<nb_heads, value<embedding_dim,
            tag3<SUBNET>>>>>>>>>>>>>>>>>>>>;

        template <long embedding_dim, typename SUBNET>
        using feed_forward =
            add_prev5<
            linear<embedding_dim,
            dropout_rate<10, gelu<linear<embedding_dim * 4,
            tag5<SUBNET>>>>>>;

        template <typename SUBNET>
        using transformer_block =
            feed_forward<embedding_size,
            rms_norm<multihead_attention<embedding_size, number_of_heads,
            rms_norm<SUBNET>>>>;
    }

    template <long num_filters, long x_stride, long y_stride, typename SUBNET>
    using comp = con<num_filters, x_stride, y_stride, x_stride, y_stride, SUBNET>;

    // Transformer-based Large Language Model (LLM) architecture

    /**
     * Transformer-based Large Language Model (LLM) architecture version 1.0
     *
     * This network defines a Transformer-style LLM inspired by the original
     * "Attention Is All You Need" (Vaswani et al., 2017) architecture
     * with some modern optimizations:
     *
     * 1. Input Layer:
     *    - Uses a matrix of integers (input<matrix<int, 0, 1>>)
     *    - Represents tokenized text as integer sequences
     *
     * 2. Embeddings:
     *    - Combines token embeddings with positional encodings
     *    - Uses tanh activation (positional_embeddings<...>)
     *
     * 3. Transformer Blocks:
     *    - Stacks 'number_of_blocks' identical blocks (repeat<number_of_blocks, ...>)
     *    - Each block (v1_0_6::transformer_block) contains:
     *      a. RMS Normalization: For training stability
     *      b. Multi-head Attention: With causal masking for autoregressive tasks
     *      c. Feed-forward Network: With GELU activation
     *
     * 4. Classification Head:
     *    - For next-token prediction (classification_head<vocab_size, ...>)
     *
     * Key features:
     * - Uses 1x1 convolutions for Q, K, V projections in the attention mechanism
     * - Implements residual connections throughout the network
     * - Employs dropout for regularization
     *
     * Global parameters used:
     * - vocab_size: Size of the vocabulary
     * - number_of_blocks: Number of Transformer blocks
     * - embedding_size: Dimension of the embeddings
     *
     * This architecture balances classic Transformer elements with modern
     * optimizations, making it suitable for various language modeling tasks.
     */
    using net_v1_0 = classification_head<vocab_size,
        repeat<number_of_blocks, v1_0_6::transformer_block,
        positional_embeddings<vocab_size, embedding_size,
        input<matrix<int, 0, 1>>>>>;

    /**
     * Advanced Transformer-based Large Language Model (LLM) architecture version 1.1
     *
     * This network defines an enhanced version of the Transformer-style LLM,
     * featuring input compression for improved efficiency and local context awareness:
     *
     * 1. Input Layer:
     *    - Uses a matrix of integers (input<matrix<int, 0, 1>>)
     *    - Represents tokenized text as integer sequences
     *
     * 2. Embeddings:
     *    - Combines token embeddings with positional encodings (positional_embeddings<...>)
     *
     * 3. Input Compression:
     *    - Two-stage compression using convolutional layers (llm::comp<...>)
     *    a. First compression: Reduces sequence length
     *       Parameters: number_of_heads filters, stride 2 in sequence dimension
     *    b. Second compression: Reduces embedding dimension
     *       Parameters: 1 filter, stride 1 in both dimensions
     *    - Captures local relationships and reduces input size for efficiency
     *
     * 4. Transformer Blocks:
     *    - Stacks 'number_of_blocks' identical blocks (repeat<number_of_blocks, ...>)
     *    - Each block (v1_1_4::transformer_block) contains:
     *      a. RMS Normalization: For improved training stability
     *      b. Enhanced Multi-head Attention:
     *         - Uses linear layers without bias for Q, K, V projections
     *         - Implements efficient head splitting and stacking
     *      c. Optimized Feed-forward Network: With GELU activation
     *
     * 5. Classification Head:
     *    - For next-token prediction (classification_head<vocab_size, ...>)
     *
     * Key improvements over v1_0:
     * - Allows for longer input sequences and larger initial embeddings
     * - Uses convolutional compression to maintain computational efficiency
     * - Captures local context through input convolutions
     * - Refines attention mechanism for potential performance gains
     * - Balances increased model capacity with efficient processing
     *
     * Global parameters used:
     * - vocab_size: Size of the vocabulary
     * - number_of_blocks: Number of Transformer blocks
     * - embedding_size: Final dimension of the embeddings
     * - number_of_heads: Number of attention heads, also used in compression
     *
     * This architecture enhances the v1_0 model by allowing for richer input
     * representations while maintaining computational efficiency through
     * strategic compression, potentially improving both performance and
     * the model's ability to capture local and global contexts.
     */
    using net_v1_1 = classification_head<vocab_size,
        repeat<number_of_blocks, v1_1_4::transformer_block,
        //llm::comp<1, 1, 1, llm::comp<llm::number_of_heads, 2, 1,
        positional_embeddings<vocab_size, embedding_size,
        input<matrix<int, 0, 1>>>>>;
}

#endif // LlmNet_H