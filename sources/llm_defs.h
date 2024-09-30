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
    const int vocab_size = 20000;                                           // Size of the vocabulary
    const int sequence_size = 32;                                           // Length of the sequence
    const int number_of_heads = 6;                                          // Number of attention heads
    const int number_of_blocks = 2;                                         // Number of transformer blocks
    const int embedding_size = (36 / number_of_heads) * number_of_heads;    // Size of the embedding

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
    template <int num_logits, typename SUBNET>
    using classification_head = loss_multiclass_log<fc<num_logits, SUBNET>>;

    namespace v1_0_6 {
        // Basic layers for "Query", "Key", and "Value"
        // These are 1x1 convolutions that project the input into different spaces
        template <int nb_heads, int num_filters_out, typename SUBNET>
        using query = con<nb_heads, 1, 1, 1, nb_heads, SUBNET>;

        template <int nb_heads, int num_filters_out, typename SUBNET>
        using key = con<nb_heads, 1, 1, 1, nb_heads, SUBNET>;

        template <int nb_heads, int num_filters_out, typename SUBNET>
        using value = con<nb_heads, 1, 1, 1, nb_heads, SUBNET>;

        // Core masked multihead attention block
        // This implements the core of the transformer's attention mechanism
        template <int embedding_dim, int nb_heads, typename SUBNET>
        using multihead_attention =
            add_prev3<  // Add the output to the input (residual connection)
            cont<1, 1, nb_heads, 1, nb_heads,  // Combine heads
            multm_prev1<  // Matrix multiplication with value
            dropout_rate<10,  // Apply dropout
            softmaxm<  // Apply softmax to attention scores
            tril_mask<  // Apply triangular mask for causal attention
            scale_weights<nb_heads, embedding_dim,  // Scale dot products
            multm_prev2<  // Matrix multiplication of query and key
            query<nb_heads, embedding_dim, skip3<  // Query transformation
            tag2<transpose<  // Transpose key for dot product
            key<nb_heads, embedding_dim, skip3<  // Key transformation
            tag1<value<nb_heads, embedding_dim,  // Value transformation
            tag3<SUBNET>>>>>>>>>>>>>>>>>;

        // Feedforward block
        // This implements the position-wise feedforward network in the transformer
        template <int embedding_dim, typename SUBNET>
        using feed_forward =
            add_prev5<  // Add the output to the input (residual connection)
            linear<embedding_size,  // Project back to embedding size
            dropout_rate<10,  // Apply dropout
            gelu<linear<embedding_size * 4,  // Expand and apply GELU activation
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
        template <int num_filters_out, typename SUBNET>
        using query = linear_no_bias<num_filters_out, SUBNET>;
        template <int num_filters_out, typename SUBNET>
        using key = linear_no_bias<num_filters_out, SUBNET>;
        template <int num_filters_out, typename SUBNET>
        using value = linear_no_bias<num_filters_out, SUBNET>;

        template <int embedding_dim, int nb_heads, typename SUBNET>
        using multihead_attention =
            add_prev3<
            linear<embedding_dim,
            hstack<
            multm_prev1<
            dropout_rate<10, softmaxm<
            tril_mask<
            scale_weights<nb_heads, embedding_dim,
            multm_prev2<
            hsplit<nb_heads, query<embedding_dim, skip3<
            tag2<transpose<hsplit<nb_heads, key<embedding_dim, skip3<
            tag1<hsplit<nb_heads, value<embedding_dim,
            tag3<SUBNET>>>>>>>>>>>>>>>>>>>>>;

        template <int embedding_dim, typename SUBNET>
        using feed_forward =
            add_prev5<
            linear<embedding_size,
            dropout_rate<10, gelu<linear<embedding_size * 4,
            tag5<SUBNET>>>>>>;

        template <typename SUBNET>
        using transformer_block =
            feed_forward<embedding_size,
            multihead_attention<embedding_size, number_of_heads,
            rms_norm<SUBNET>>>;
    }

    // VSLM networks
    using net_v1_0 = classification_head<vocab_size,
        repeat<number_of_blocks, v1_0_6::transformer_block,
        positional_embeddings<vocab_size, embedding_size,
        input<matrix<int, 0, 1>>>>>;

    using net_v1_1 = classification_head<vocab_size,
        repeat<number_of_blocks, v1_1_4::transformer_block,
        positional_embeddings<vocab_size, embedding_size,
        input<matrix<int, 0, 1>>>>>;

    // Feedforward block
    /*template <int embedding_dim, typename SUBNET>
    using feed_forward =
        add_prev5<
        linear<embedding_size,
        dropout_rate<10, gelu<linear<embedding_size * 4,
        tag5<SUBNET>>>>>>;*/
    /*template <int sequence_dim, int embedding_dim, typename SUBNET>
    using feed_forward =
        add_prev5<
        scale5<con<1, 1, 1, 1, 1,
        //cont<1, sequence_dim, embedding_dim, sequence_dim, embedding_dim,
        sig<fc<embedding_size,
        dropout_rate<10, gelu<bn_fc<fc<embedding_size / 4,
        avg_pool_everything<tag5<SUBNET>>>>>>>>>>>;*/
        /*using feed_forward =
            add_prev5<
            cont<1, sequence_dim, embedding_dim, sequence_dim, embedding_dim,
            fc<embedding_size,
            dropout_rate<10, gelu<bn_fc<fc<embedding_size * 2,
            tag5<SUBNET>>>>>>>>;*/
}

#endif // LlmNet_H