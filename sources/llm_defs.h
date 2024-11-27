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
 * Key features:
 * - RMS normalization for enhanced stability
 * - Optimized residual connections
 * - Causal masking for autoregressive attention
 */

#include "dlib_ext.h"
#include <dlib/dnn.h>

namespace transformer
{
    using namespace dlib;

    // Scale Weights Layer
    template <long d_k_>
    class scale_weights_ : public multiply_ {
    public:
        explicit scale_weights_() : multiply_(1.0f / std::sqrt(static_cast<float>(d_k_))) {}
    };

    template <long d_k, typename SUBNET>
    using scale_weights = add_layer<scale_weights_<d_k>, SUBNET>;

    namespace def {
        template <long seq_len, long d_model, typename SUBNET>
        using query = extract<0, 1, seq_len, d_model, SUBNET>;

        template <long seq_len, long d_model, typename SUBNET>
        using key = extract<seq_len* d_model, 1, seq_len, d_model, SUBNET>;

        template <long seq_len, long d_model, typename SUBNET>
        using value = extract<(seq_len* d_model) * 2, 1, seq_len, d_model, SUBNET>;

        /**
         * Multi-Head Attention Implementation
         *
         * Models selective attention and parallel processing:
         * 1. Initial Processing:
         *    - RMS normalization to stabilize neural activation
         *    - Linear projection to (Q,K,V) space for feature extraction
         *    - Split into distinct Q, K, V tensors
         * 2. Attention Head Formation:
         *    - Split into num_heads parallel processors
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
         *    - 10% output & attention weights
         *    - Inference version: dropout -> multiply
         *
         * Template Parameters:
         * @param seq_len: Maximum sequence length
         * @param d_model: Model dimension
         * @param num_heads: Number of attention heads
         * @param SUBNET: Input subnet type
         */
        template <template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET>
        using multihead_attention =
            add_prev1<
            DO<linear<d_model,
            hstack<
            multm_prev3<
            DO<softmaxm<tril_mask<
            scale_weights<d_model / num_heads,
            multm_prev4<hsplit<num_heads, query<seq_len, d_model, skip2<
            tag4<transpose<hsplit<num_heads, key<seq_len, d_model, skip2<
            tag3<hsplit<num_heads, value<seq_len, d_model,
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
         *    - 10% dropout during training
         *    - add_prev5 residual connection
         * Architectural modifications:
         * - RMS norm for stability
         * - GELU for smooth non-linearity
         * - Optimized residual connections
         *
         * @param d_model: Model dimension
         * @param SUBNET: Input subnet type
         */
        template <template <typename> class ACT, template <typename> class DO, long d_model, typename SUBNET>
        using feed_forward =
            add_prev5<
            DO<linear<d_model, ACT<linear<d_model * 4, rms_norm<
            tag5<SUBNET>>>>>>>;

        template <template <typename> class ACT, template <typename> class DO, long seq_len, long d_model, long num_heads, typename SUBNET>
        using transformer =
            feed_forward<ACT, DO, d_model,
            multihead_attention<ACT, DO, seq_len, d_model, num_heads, SUBNET>>;
    }

    // Positional Embeddings
    template <long num_embeddings, long embedding_length, typename SUBNET>
    using positional_embeddings = positional_encodings<embeddings<num_embeddings, embedding_length, SUBNET>>;

    // Learned Positional Embeddings
    template <long num_embeddings, long embedding_length, typename SUBNET>
    using learned_positional_embeddings = add_prev9<linear<embedding_length, skip10<
        tag9<embeddings<num_embeddings, embedding_length, tag10<SUBNET>>>>>>;

    // Classification Head
    template <long num_logits, long embedding_length, typename SUBNET>
    using classification_head = loss_cross_entropy<linear<num_logits, rms_norm<SUBNET>>>;
    
    template <template <typename> class ACT, long embedding_length, typename SUBNET>
    using squeezing = fc<embedding_length / 4, ACT<fc<embedding_length / 8, SUBNET>>>;
    
    template <template <typename> class ACT, long num_logits, long embedding_length, typename SUBNET>
    using classification_head_fc = loss_multiclass_log<fc<num_logits, squeezing<ACT, embedding_length, rms_norm<SUBNET>>>>;

    template <typename SUBNET>
    using dropout_10 = dropout_rate<10, SUBNET>;

    /**
     * @brief Transformer model configuration template
     *
     * Provides a flexible and type-safe configuration mechanism for Transformer models
     * with compile-time parameter validation and network generation.
     *
     * @tparam vocab_size Vocabulary size for token embedding
     * @tparam num_layers Number of Transformer layers
     * @tparam num_heads Number of attention heads
     * @tparam embedding_dim Dimension of token embeddings
     * @tparam max_seq_len Maximum sequence length
     * @tparam activation_func Activation function type
     * @tparam dropout_policy Dropout regularization policy
     */
    template <
        long vocab_size = 2000,                                 // Default vocabulary size
        long num_layers = 6,                                    // Default number of layers
        long num_heads = 4,                                     // Default number of attention heads
        long embedding_dim = 256,                               // Default embedding dimension
        long max_seq_len = 80,                                  // Default maximum sequence length
        template <typename> class activation_func = gelu,      // Default activation function
        template <typename> class dropout_policy = dropout_10  // Default dropout policy
    >
    struct transformer_config {
        // Core model parameters
        static constexpr long VOCAB_SIZE = vocab_size;
        static constexpr long NUM_LAYERS = num_layers;
        static constexpr long NUM_HEADS = num_heads;
        static constexpr long EMBEDDING_DIM = embedding_dim;
        static constexpr long MAX_SEQ_LEN = max_seq_len;

        // Derived and calculated parameters
        static constexpr long HEAD_DIM = EMBEDDING_DIM / NUM_HEADS;
        static constexpr long FFN_HIDDEN_DIM = EMBEDDING_DIM * 4;

        /**
         * @brief Compile-time validation of model configuration
         *
         * Performs static assertions to ensure valid model parameters
         */
        struct validation {
            static_assert(VOCAB_SIZE > 0, "Vocabulary size must be positive");
            static_assert(NUM_LAYERS > 0, "Number of layers must be positive");
            static_assert(NUM_HEADS > 0, "Number of attention heads must be positive");
            static_assert(EMBEDDING_DIM % NUM_HEADS == 0, "Embedding dimension must be divisible by number of heads");
        };

        /**
         * @brief Network type generation based on training/inference mode
         *
         * Generates different network types for training and inference
         * using the configured parameters
         *
         * @tparam is_training Determines training or inference network type
         */
        template <typename SUBNET>
        using t_transformer_block = def::transformer<activation_func, dropout_policy, MAX_SEQ_LEN, EMBEDDING_DIM, NUM_HEADS, SUBNET>;
        template <typename SUBNET>
        using i_transformer_block = def::transformer<activation_func, multiply, MAX_SEQ_LEN, EMBEDDING_DIM, NUM_HEADS, SUBNET>;

        template<bool is_training>
        using network_type = std::conditional_t<is_training,
            classification_head_fc<activation_func, VOCAB_SIZE, EMBEDDING_DIM,
            repeat<NUM_LAYERS, t_transformer_block,
            positional_embeddings<VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>,
            classification_head_fc<activation_func, VOCAB_SIZE, EMBEDDING_DIM,
            repeat<NUM_LAYERS, i_transformer_block,
            positional_embeddings<VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>
        >;

        /**
         * @brief Model configuration information and debugging utility
         *
         * Provides methods to generate human-readable model configuration details
         */
        struct model_info {
            /**
             * @brief Generate a detailed description of the model configuration
             *
             * @return String containing model configuration details
             */
            static std::string describe() {
                std::stringstream ss;
                ss << "transformer model configuration:\n"
                    << "- vocabulary size: " << VOCAB_SIZE << "\n"
                    << "- layers: " << NUM_LAYERS << "\n"
                    << "- attention heads: " << NUM_HEADS << "\n"
                    << "- embedding dimension: " << EMBEDDING_DIM << "\n"
                    << "- max sequence length: " << MAX_SEQ_LEN;
                return ss.str();
            }
        };
    };

    using vslm = transformer_config<>; // Very Small Language Model

    /**
     * @example Configuration and Usage Examples
     *
     * // Creating different transformer configurations
     * using default_transformer = transformer_config<>;
     * using large_transformer = transformer_config<
     *     5000,   // Larger vocabulary
     *     8,      // More layers
     *     8,      // More heads
     *     512,    // Larger embedding dimension
     *     128     // Longer sequences
     * >;
     *
     * // Network type instantiations for different modes
     * using train_network = default_transformer::network_type<true>;
     * using inference_network = default_transformer::network_type<false>;
     *
     * // Utility function to print model configuration
     * void print_model_info() {
     *     std::cout << default_transformer::model_info::describe() << std::endl;
     * }
     *
     * @note
     * - Supports compile-time configuration
     * - Provides static validation of model parameters
     * - Enables dynamic network type generation
     * - Offers advanced hyperparameter tuning utilities
     *
     * @author Cydral
     * @site https://github.com/Cydral
     * @version 1.0
     * @date 11/2024
     */
}

#endif // LlmNet_H
