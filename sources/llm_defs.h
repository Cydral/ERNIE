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
    const long vocab_size = 80000;           // Size of the vocabulary - Default: en-fr.40k
    const long number_of_blocks = 4;         // Number of transformer blocks - Default: 4
    const long number_of_heads = 8;          // Number of attention heads - Default: 8
    const long embedding_size = 64;         // Size of the embedding (d_model) - Default: 64
    const long sequence_size = 24;           // Maximum length of the input sequence - Default: 24

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

    // Classification head
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
     * - RMS normalization for training stability
     *
     * This compact design balances model capacity with efficiency, making it
     * suitable for learning from and generating short texts.
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