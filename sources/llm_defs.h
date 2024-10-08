#ifndef LlmNet_H
#define LlmNet_H

/**
 * @file llm_defs.h
 * @brief Definitions for a small Transformer-based Language Model using Dlib
 *
 * This file contains the definitions, constants, and parameters for implementing
 * a compact Transformer model suitable for learning short texts and answering
 * simple questions. It utilizes the Dlib library and defines the structure of
 * the LLM network, including vocabulary size, sequence length, number of
 * attention heads, Transformer blocks, and embedding size.
 *
 * @note These parameters are tailored for a small LLM and can be adjusted based
 * on specific requirements and available computational resources.
 */

//#include <dlib/dnn.h>
#include "dlib_ext.h"

namespace llm
{
    using namespace dlib;

    // Global parameters for the small Transformer network
    const long vocab_size = 12000;           // Size of the vocabulary
    const long number_of_blocks = 2;         // Number of transformer blocks
    const long number_of_heads = 16;         // Number of attention heads
    const long embedding_size = 256;         // Size of the embedding (d_model)
    const long sequence_size = 64;           // Maximum length of the input sequence

    // Scale Weights Layer
    template <long d_k_>
    class scale_weights_ : public multiply_ {
        static_assert(d_k_ > 0, "The dimension of each attention head must be > 0");

    public:
        explicit scale_weights_() : multiply_(1.0f / std::sqrt(static_cast<float>(d_k_))) {}
    };

    template <long d_k, typename SUBNET>
    using scale_weights = add_layer<scale_weights_<d_k>, SUBNET>;

    // Positional Embeddings
    template <unsigned long nb_embeddings, unsigned long embedding_length, typename SUBNET>
    using positional_embeddings =
        positional_encodings<
        embeddings<nb_embeddings, embedding_length, SUBNET>>;

    // Classification head
    template <long num_logits, typename SUBNET>
    using classification_head = loss_multiclass_log<fc<num_logits, SUBNET>>;

    /*
        Selector Layer

        This custom layer, termed "Selector", is designed to enhance the model's ability to capture
        and process complex relationships between words and their vector representations before
        the final classification stage. It combines principles from convolutional neural networks (CNNs)
        and fully connected networks to create a powerful feature extraction and integration mechanism.

        Key components and their rationale:
        1. Convolutional layers:
           - Capture local patterns and relationships between adjacent words or features.
           - Hierarchical feature extraction with increasing number of filters (nb_heads to nb_heads*2).
        2. Max pooling:
           - Reduces dimensionality while retaining the most salient features.
           - Introduces a degree of translational invariance.
        3. Fully connected layers:
           - Enable global feature integration and complex non-linear mappings.
           - Expansion (d_model*4) followed by contraction (d_model*2) allows for rich feature interaction.
        4. Activation functions (ReLU):
           - Introduce non-linearity, crucial for learning complex patterns.
           - Help mitigate the vanishing gradient problem.
        5. Batch normalization:
           - Stabilizes learning by normalizing layer inputs.
           - Can accelerate training and improve generalization.

        This architecture draws inspiration from successful CNN models in computer vision, adapting
        the concept to NLP tasks. It aims to provide a final layer of abstraction and feature
        integration before classification, potentially improving the model's ability to discern
        subtle patterns and relationships in the input data.
        The "Selector" name aptly describes its function: it selects and emphasizes the most
        relevant features and relationships for the downstream task, acting as a sophisticated
        feature selector and integrator.
    
        This layer is intended to be used as a termination layer before the final classification
        in transformer-based or other deep learning models for NLP tasks. It can be particularly
        useful in scenarios where capturing both local and global relationships between words
        and their vector representations is crucial for the task at hand.
    */
    template <long d_model, long nb_heads, typename SUBNET>
    using selectors = relu<fc<d_model * 2, relu<bn_fc<fc<d_model * 4,
        max_pool<2, 2, 2, 2, relu<con<nb_heads * 2, 5, 5, 1, 1,
        max_pool<3, 3, 2, 2, relu<con<nb_heads, 7, 7, 1, 1,
        SUBNET>>>>>>>>>>>;

    namespace v1_1_4 {
        template <long seq_len, long d_model, typename SUBNET>
        using query = extract<0, 1, seq_len, d_model, SUBNET>;

        template <long seq_len, long d_model, typename SUBNET>
        using key = extract<seq_len * d_model, 1, seq_len, d_model, SUBNET>;

        template <long seq_len, long d_model, typename SUBNET>
        using value = extract<(seq_len * d_model) * 2, 1, seq_len, d_model, SUBNET>;

        template <long seq_len, long d_model, long nb_heads, typename SUBNET>
        using multihead_attention =
            add_prev1<
            linear_no_bias<d_model,
            hstack<
            multm_prev3<
            softmaxm<tril_mask<
            scale_weights<d_model / nb_heads,
            multm_prev4<hsplit<nb_heads, query<seq_len, d_model, skip2<
            tag4<transpose<hsplit<nb_heads, key<seq_len, d_model, skip2<
            tag3<hsplit<nb_heads, value<seq_len, d_model,
            tag2<hsplit<3, linear_no_bias<d_model * 3,            
            tag1<SUBNET>>>>>>>>>>>>>>>>>>>>>>>;

        template <long d_model, typename SUBNET>
        using feed_forward =
            add_prev5<            
            linear<d_model, gelu<linear<d_model * 4,            
            tag5<SUBNET>>>>>;

        template <long seq_len, long d_model, long nb_heads, typename SUBNET>
        using transformer =
            feed_forward<d_model,
            multihead_attention<seq_len, d_model, nb_heads, 
            rms_norm<SUBNET>>>;
    }

    /**
     * Compact Transformer-based Language Model architecture
     *
     * This network defines a small Transformer-style LLM suitable for learning
     * short texts and answering simple questions. Key features include:
     *
     * 1. Input Layer: Accepts tokenized text as integer sequences
     * 2. Embeddings: Combines token embeddings with positional encodings
     * 3. Transformer Blocks: Stacks multiple identical blocks, each containing:
     *    - RMS Normalization
     *    - Multi-head Attention with causal masking
     *    - Feed-forward Network with GELU activation
     * 4. Classification Head: For next-token prediction
     *
     * The architecture incorporates:
     * - Residual connections
     * - Dropout for regularization
     * - RMS normalization for training stability
     *
     * This compact design balances model capacity with efficiency, making it
     * suitable for learning from and generating short texts.
     */    
    template <typename SUBNET>
    using transformer_block = v1_1_4::transformer<sequence_size, embedding_size, number_of_heads, SUBNET>;

    using net_v1_1 = classification_head<vocab_size,        
        selectors<embedding_size, number_of_heads,
        repeat<number_of_blocks, transformer_block,
        positional_embeddings<vocab_size, embedding_size,
        input<matrix<int, 0, 1>>>>>>;
}

#endif // LlmNet_H