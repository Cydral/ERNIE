#ifndef DlibExt_H
#define DlibExt_H

#include <iostream>
#include <dlib/dnn.h>
#include <dlib/matrix.h>
#include <dlib/matrix/matrix_utilities.h>
#include <dlib/data_io.h>

using namespace std;
using namespace dlib;

#ifdef DLIB_USE_CUDA
#include "cuda_dlib_ext.cuh"
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cudnn.h>
#endif // DLIB_USE_CUDA

#define DLIB_TEST_MSG(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "test failed: " << msg << std::endl; \
        } else { \
            std::cout << "test passed: " << msg << std::endl; \
        } \
    } while (0)

volatile std::sig_atomic_t g_interrupt_signal_received = 0;
void signalHandler(int signal) {
    if (signal == SIGINT) {
        g_interrupt_signal_received = 1;
        cout << "\ninterrupt detected (CTRL+C), cleaning up and closing the program" << endl;
    }
}

#ifdef DLIB_USE_CUDA
static const char* cudnn_get_error_string(cudnnStatus_t s)
{
    switch (s)
    {
    case CUDNN_STATUS_NOT_INITIALIZED:
        return "CUDA Runtime API initialization failed.";
    case CUDNN_STATUS_ALLOC_FAILED:
        return "CUDA Resources could not be allocated.";
    case CUDNN_STATUS_BAD_PARAM:
        return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_EXECUTION_FAILED:
        return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
        return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_ARCH_MISMATCH:
        return "CUDNN_STATUS_ARCH_MISMATCH: Your GPU is too old and not supported by cuDNN";
    default:
        return "A call to cuDNN failed";
    }
}

#define CHECK_CUDNN(call) \
do{ \
    const cudnnStatus_t error = call; \
    if (error != CUDNN_STATUS_SUCCESS) \
    { \
        std::ostringstream sout; \
        sout << "Error while calling " << #call << " in file " << __FILE__ << ":" << __LINE__ << ". "; \
        sout << "code: " << error << ", reason: " << cudnn_get_error_string(error); \
        throw dlib::cudnn_error(sout.str()); \
    } \
}while(false)

static const char* cublas_get_error_string(cublasStatus_t s)
{
    switch (s)
    {
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUDA Runtime API initialization failed.";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUDA Resources could not be allocated.";
    default:
        return "A call to cuBLAS failed";
    }
}

#define CHECK_CUBLAS(call) \
do{ \
    const cublasStatus_t error = call; \
    if (error != CUBLAS_STATUS_SUCCESS) \
    { \
        std::ostringstream sout; \
        sout << "Error while calling " << #call << " in file " << __FILE__ << ":" << __LINE__ << ". "; \
        sout << "code: " << error << ", reason: " << cublas_get_error_string(error); \
        throw dlib::cublas_error(sout.str()); \
    } \
}while(false)
#endif // DLIB_USE_CUDA

void DBG_INFO(std::string dbg_msg) {
    if (!dbg_msg.empty()) cout << dbg_msg << endl;
}
void DBG_INFO(std::string dbg_msg, const tensor& t, const bool display_t = false, int S = 10, int K = 5, int R = 8, int C = 8) {
    if (!dbg_msg.empty()) {
        cout << dbg_msg << "num_samples=" << t.num_samples() << ", k=" << t.k() << ", nr=" << t.nr() << ", nc=" << t.nc() << endl;
        if (display_t) {
            S = std::min(K, static_cast<int>(t.num_samples()));
            K = std::min(K, static_cast<int>(t.k()));
            R = std::min(R, static_cast<int>(t.nr()));
            C = std::min(C, static_cast<int>(t.nc()));
            for (int s = 0; s < t.num_samples(); ++s) {
                cout << "[";
                for (int k = 0; k < t.k(); ++k) {
                    cout << "[\t";
                    for (int r = 0; r < t.nr(); ++r) {
                        for (int c = 0; c < t.nc(); ++c) {
                            if (c < C) cout << setw(8) << fixed << setprecision(3) << t.host()[tensor_index(t, s, k, r, c)] << " ";
                            else if (c == C) {
                                cout << "...";
                                break;
                            }
                        }
                        if (r < R) cout << endl << "\t";
                        else if (r == R) {
                            cout << endl << "(...)" << endl;
                            break;
                        }
                    }
                    cout << "]";
                }
                if (s < S) cout << "]" << endl;
                if (s == (S - 1)) break;
            }
        }
    }
}

namespace densenet
{
    using namespace dlib;
    template <template <typename> class ACT, template <typename> class BN, long k>
    struct def {
        template <long num_filters, long ks, int s, typename SUBNET>
        using conp = add_layer<con_<num_filters, ks, ks, s, s, ks / 2, ks / 2>, SUBNET>;

        template <typename INPUT>
        using stem = add_layer<max_pool_<3, 3, 2, 2, 1, 1>, ACT<BN<conp<2 * k, 7, 2, INPUT>>>>;

        template <long num_filters, typename SUBNET>
        using transition = avg_pool<2, 2, 2, 2, con<num_filters, 1, 1, 1, 1, ACT<BN<SUBNET>>>>;

        template <typename SUBNET>
        using dense_layer = concat2<tag1, tag2,
            tag2<conp<k, 3, 1,
            ACT<BN<conp<4 * k, 1, 1,
            ACT<BN<tag1<SUBNET>>>>>>>>>;

        template <size_t n4, size_t n3, size_t n2, size_t n1, typename INPUT>
        using backbone = ACT<BN<
            repeat<n4, dense_layer, transition<k* (2 + n1 + 2 * n2 + 4 * n3) / 8,
            repeat<n3, dense_layer, transition<k* (2 + n1 + 2 * n2) / 4,
            repeat<n2, dense_layer, transition<k* (2 + n1) / 2,
            repeat<n1, dense_layer, stem<INPUT>>>>>>>>>>;
    };
}

namespace dlib {
    class compressed_float {
    private:
        static constexpr int MANTISSA_BITS = 7;
        static constexpr int EXP_BITS = 8;

        static inline bool compression_enabled = false;

    public:
        static void enable_compression() { compression_enabled = true; }
        static void disable_compression() { compression_enabled = false; }
        static bool is_compression_enabled() { return compression_enabled; }

        static uint16_t compress(float value) {
            if (value == 0.0f) return 0;

            uint32_t raw;
            std::memcpy(&raw, &value, sizeof(float));

            uint32_t sign = (raw >> 31) & 0x1;
            uint32_t exp = ((raw >> 23) & 0xFF);
            uint32_t mantissa = (raw & 0x7FFFFF);

            uint16_t compressed = (sign << 15) |
                ((exp & ((1 << EXP_BITS) - 1)) << MANTISSA_BITS) |
                (mantissa >> (23 - MANTISSA_BITS));

            return compressed;
        }

        static float decompress(uint16_t compressed) {
            if (compressed == 0) return 0.0f;

            uint32_t sign = (compressed >> 15) & 0x1;
            uint32_t exp = (compressed >> MANTISSA_BITS) & ((1 << EXP_BITS) - 1);
            uint32_t mantissa = (compressed & ((1 << MANTISSA_BITS) - 1));

            // IEEE-754 float
            uint32_t raw = (sign << 31) | (exp << 23) | (mantissa << (23 - MANTISSA_BITS));

            float result;
            std::memcpy(&result, &raw, sizeof(float));
            return result;
        }
    };

    template<typename T>
    typename std::enable_if<std::is_same<T, float>::value, std::ostream&>::type
        operator<<(std::ostream& out, const T& item) {
        if (compressed_float::is_compression_enabled()) {
            uint16_t compressed = compressed_float::compress(item);
            out.write(reinterpret_cast<const char*>(&compressed), sizeof(uint16_t));
        }
        else {
            out.write(reinterpret_cast<const char*>(&item), sizeof(float));
        }
        return out;
    }

    template<typename T>
    typename std::enable_if<std::is_same<T, float>::value, std::istream&>::type
        operator>>(std::istream& in, T& item) {
        if (compressed_float::is_compression_enabled()) {
            uint16_t compressed = 0;
            in.read(reinterpret_cast<char*>(&compressed), sizeof(uint16_t));
            item = compressed_float::decompress(compressed);
        }
        else {
            in.read(reinterpret_cast<char*>(&item), sizeof(float));
        }
        return in;
    }

    namespace tt {
        enum class operation_mode { CHANNEL_WISE = 0, PLANE_WISE = 1 };

        /* TO BE ADDED TO <tensor_tools.h> */
        // -----------------------------------------------------------------------------------
        void embeddings(
            resizable_tensor& dest,
            const tensor& src,
            const tensor& embs
        );
        /*!
            requires
                - src.nr() > 0
                - embs.num_samples() > 0
                - embs.k() > 0
                - embs.nr() == 1
                - embs.nc() == 1
                - dest.num_samples() == src.num_samples()
                - dest.k() == src.k()
                - dest.nr() == src.nr()
                - dest.nc() == embs.k()
            ensures
                - Projects tokens from the input tensor `src` into embeddings stored in `embs`.
                - The resulting embeddings are stored in the `dest` tensor.
                - For all valid s (0 <= s < dest.num_samples()),
                               k (0 <= k < dest.k()),
                               r (0 <= r < dest.nr()),
                               c (0 <= c < dest.nc()):
                    - Let token_idx = static_cast<unsigned long>(src(s,k,r,0))
                    - If token_idx < embs.num_samples():
                        - #dest(s,k,r,c) == embs(token_idx, c, 0, 0)
                    - Else:
                        - #dest(s,k,r,c) == 1
                - The function iterates over all elements of src and populates dest accordingly.
                - If a token index in src is out of range (>= embs.num_samples()),
                  the corresponding embedding in dest is filled with 1's instead of 0's.
        */

        void embeddings_gradient(
            const tensor& prev,
            const tensor& gradient_input,
            tensor& grads,
            const tensor& freqs,
            float learning_rate,
            bool scale
        );
        /*!
            requires
                - prev.nr() > 0
                - gradient_input.num_samples() == prev.num_samples()
                - gradient_input.k() == prev.k()
                - gradient_input.nr() == prev.nr()
                - gradient_input.nc() == grads.k()
                - grads.num_samples() > 0
                - grads.k() > 0
                - grads.nr() == 1
                - grads.nc() == 1
            ensures
                - Updates the `grads` tensor based on the gradients in `gradient_input`.
                - For each sample s, channel k, and row r in prev:
                    - Retrieves the token index from prev[s,k,r]
                    - If the token index is valid (< grads.num_samples()):
                        - If scale is true:
                            - Computes a frequency scale factor based on freqs[token_idx]
                            - The scale factor is min(0.15, max(2.0 * (1.0 / (1.0 + freqs[token_idx])), 1.0))
                        - For each column c in gradient_input:
                            - Updates grads[token_idx, c] -= gradient_input[s,k,r,c] * rate * freq_scale
                - The updates to grads are performed atomically to handle concurrent updates to the same embedding.
                - The function is thread-safe and processes samples in parallel.
        */

        void rms_normalize(
            const double eps,
            resizable_tensor& dest,
            resizable_tensor& scale,
            const tensor& src,
            const tensor& gamma
        );
        /*!
            requires
                - eps > 0
                - gamma.k() == src.k()
                - gamma.nr() == 1
                - gamma.nc() == 1
            ensures
                - have_same_dimensions(#dest, src) == true
                - #scale.size() == src.num_samples()
                - #dest == the RMS normalized version of src
                - #scale contains the RMS (Root Mean Square) values used to normalize each sample of src.
                - Each element of #dest is computed as:
                    - #dest[n, k, i, j] == src[n, k, i, j] * gamma[k] / scale[n]
                where n is the sample index, k is the channel index, and i, j are the spatial indices.
        !*/

        void rms_normalize_gradient(
            const tensor& gradient_input,
            const tensor& scale,
            const tensor& src,
            const tensor& gamma,
            tensor& src_grad,
            tensor& gamma_grad,
            resizable_tensor& dscale
        );
        /*!
            requires
                - scale.size() == src.num_samples()
                - have_same_dimensions(gamma, gamma_grad)
                - gamma.k() == src.k()
                - gamma.nr() == 1
                - gamma.nc() == 1
                - have_same_dimensions(gradient_input, src)
                - have_same_dimensions(gradient_input, src_grad)
            ensures
                - Let f(src, gamma) == dot(gradient_input, dest output of
                  rms_normalize(eps, dest, scale, src, gamma))
                - Adds the gradient of f() with respect to src to #src_grad
                - Assigns the gradient of f() with respect to gamma to #gamma_grad
                - #dscale contains the gradients of f() with respect to the RMS values.
        !*/

        void transpose(
            bool add_to,
            tensor& dest,
            const tensor& src
        );
        /*!
            requires
                - dest.num_samples() == src.num_samples()
                - dest.k() == src.k()
                - dest.nr() == src.nc()
                - dest.nc() == src.nr()
                - is_same_object(dest, src) == false
            ensures
                - Performs a transpose operation on the nr() x nc() matrices within src.
                - If (add_to) is false:
                    - The result is stored in dest, overwriting its previous contents.
                    - For all valid n, k, r, c:
                        - #dest(n,k,c,r) == src(n,k,r,c)
                - If (add_to) is true:
                    - The result is added to the existing contents of dest.
                    - For all valid n, k, r, c:
                        - #dest(n,k,c,r) == dest(n,k,c,r) + src(n,k,r,c)
        !*/

        void split_columns(
            bool add_to,
            tensor& dest,
            const tensor& src,
            const long num_heads
        );
        /*!
            requires
                - is_same_object(dest, src) == false
                - dest.num_samples() == src.num_samples()
                - dest.k() == num_heads
                - src.k() == 1
                - dest.nr() == src.nr()
                - dest.nc() == (src.nc() / num_heads)
                - src.nc() % num_heads == 0
            ensures
                - Splits the columns of src into num_heads separate heads in dest.
                - If (add_to) is false:
                    - The result is stored in dest, overwriting its previous contents.
                    - For all valid n, h, s, d:
                        - #dest(n,h,s,d) == src(n,0,s,h*head_dim + d)
                          where head_dim = src.nc() / num_heads
                - If (add_to) is true:
                    - The result is added to the existing contents of dest.
                    - For all valid n, h, s, d:
                        - #dest(n,h,s,d) == dest(n,h,s,d) + src(n,0,s,h*head_dim + d)
                          where head_dim = src.nc() / num_heads
        !*/

        void merge_columns(
            bool add_to,
            tensor& dest,
            const tensor& src
        );
        /*!
            requires
                - is_same_object(dest, src) == false
                - dest.num_samples() == src.num_samples()
                - dest.k() == 1
                - src.k() > 1
                - dest.nr() == src.nr()
                - dest.nc() == (src.nc() * src.k())
            ensures
                - Merges the columns from separate heads in src back into a single tensor dest.
                - If (add_to) is false:
                    - The result is stored in dest, overwriting its previous contents.
                    - For all valid n, r, c:
                        - #dest(n,0,r,c) == src(n,h,r,d)
                          where h = c / src.nc() and d = c % src.nc()
                - If (add_to) is true:
                    - The result is added to the existing contents of dest.
                    - For all valid n, r, c:
                        - #dest(n,0,r,c) == dest(n,0,r,c) + src(n,h,r,d)
                          where h = c / src.nc() and d = c % src.nc()
        !*/
    }

    namespace cpu {
        /* TO BE ADDED TO <cpu_dlib.cpp> */
        namespace ttimpl
        {
            void softmax(
                const long num_locations,
                const long num_channels,
                tensor& dest,
                const tensor& src,
                tt::operation_mode mode = tt::operation_mode::CHANNEL_WISE
            )
            {
                DLIB_ASSERT(num_channels * num_locations == src.nr() * src.nc() * src.k());
                DLIB_CASSERT(have_same_dimensions(dest, src));
                const auto d = dest.host();
                const auto s = src.host();

                for (long n = 0; n < src.num_samples(); ++n)
                {
                    auto ss = s + num_locations * num_channels * n;
                    auto dd = d + num_locations * num_channels * n;

                    if (mode == tt::operation_mode::CHANNEL_WISE)
                    {
                        for (long i = 0; i < num_locations; ++i)
                        {
                            float max_val = -std::numeric_limits<float>::infinity();
                            for (long k = 0; k < num_channels; ++k)
                                max_val = std::max(max_val, ss[k * num_locations]);

                            float sum = 0.0f;
                            for (long k = 0; k < num_channels; ++k)
                            {
                                dd[k * num_locations] = std::exp(ss[k * num_locations] - max_val);
                                sum += dd[k * num_locations];
                            }
                            for (long k = 0; k < num_channels; ++k)
                                dd[k * num_locations] /= sum;

                            ++ss;
                            ++dd;
                        }
                    }
                    else if (mode == tt::operation_mode::PLANE_WISE)
                    {
                        for (long k = 0; k < num_channels; ++k)
                        {
                            auto s_channel = ss + k * num_locations;
                            auto d_channel = dd + k * num_locations;
                            for (long r = 0; r < src.nr(); ++r)
                            {
                                float max_val = -std::numeric_limits<float>::infinity();
                                for (long c = 0, idx = r * src.nc(); c < src.nc(); ++c, ++idx)
                                    max_val = std::max(max_val, s_channel[idx]);

                                if (max_val == -std::numeric_limits<float>::infinity())
                                {
                                    for (long c = 0, idx = r * src.nc(); c < src.nc(); ++c, ++idx)
                                        d_channel[idx] = 0.0f;
                                }
                                else
                                {
                                    float sum = 0.0f;
                                    for (long c = 0, idx = r * src.nc(); c < src.nc(); ++c, ++idx)
                                    {
                                        d_channel[idx] = std::exp(s_channel[idx] - max_val);
                                        sum += d_channel[idx];
                                    }
                                    for (long c = 0, idx = r * src.nc(); c < src.nc(); ++c, ++idx)
                                        d_channel[idx] /= sum;
                                }
                            }
                        }
                    }
                }
            }

            void softmax_gradient(
                const long num_locations,
                const long num_channels,
                tensor& grad,
                const tensor& dest,
                const tensor& gradient_input,
                tt::operation_mode mode = tt::operation_mode::CHANNEL_WISE
            )
            {
                DLIB_ASSERT(num_channels * num_locations == grad.nr() * grad.nc() * grad.k());
                DLIB_CASSERT(have_same_dimensions(grad, dest));
                DLIB_CASSERT(have_same_dimensions(grad, gradient_input));

                const auto d = dest.host();
                const auto g = grad.host();
                const auto in = gradient_input.host();
                for (long n = 0; n < grad.num_samples(); ++n)
                {
                    const auto d2 = d + num_locations * num_channels * n;
                    const auto g2 = g + num_locations * num_channels * n;
                    const auto in2 = in + num_locations * num_channels * n;

                    if (mode == tt::operation_mode::CHANNEL_WISE)
                    {
                        for (long i = 0; i < num_locations; ++i)
                        {
                            const auto d3 = d2 + i;
                            const auto g3 = g2 + i;
                            const auto in3 = in2 + i;
                            float sum = 0.0f;
                            for (long k = 0; k < num_channels; ++k)
                                sum += -d3[k * num_locations] * in3[k * num_locations];
                            if (is_same_object(gradient_input, grad))
                            {
                                for (long k = 0; k < num_channels; ++k)
                                    g3[k * num_locations] = d3[k * num_locations] * (sum + in3[k * num_locations]);
                            }
                            else
                            {
                                for (long k = 0; k < num_channels; ++k)
                                    g3[k * num_locations] += d3[k * num_locations] * (sum + in3[k * num_locations]);
                            }
                        }
                    }
                    else if (mode == tt::operation_mode::PLANE_WISE)
                    {
                        for (long k = 0; k < num_channels; ++k)
                        {
                            const auto d_channel = d2 + k * num_locations;
                            const auto g_channel = g2 + k * num_locations;
                            const auto in_channel = in2 + k * num_locations;
                            for (long r = 0; r < grad.nr(); ++r)
                            {
                                float sum = 0.0f;
                                for (long c = 0, idx = r * grad.nc(); c < grad.nc(); ++c, ++idx)
                                    sum += -d_channel[idx] * in_channel[idx];
                                if (is_same_object(gradient_input, grad))
                                {
                                    for (long c = 0, idx = r * grad.nc(); c < grad.nc(); ++c, ++idx)
                                        g_channel[idx] = d_channel[idx] * (sum + in_channel[idx]);
                                }
                                else
                                {
                                    for (long c = 0, idx = r * grad.nc(); c < grad.nc(); ++c, ++idx)
                                        g_channel[idx] += d_channel[idx] * (sum + in_channel[idx]);
                                }
                            }
                        }
                    }
                }
            }
        }

        void softmax(
            tensor& dest,
            const tensor& src,
            tt::operation_mode mode
        )
        {
            DLIB_CASSERT(have_same_dimensions(dest, src));
            ttimpl::softmax(src.nr() * src.nc(), src.k(), dest, src, mode);
        }

        void softmax_gradient(
            tensor& grad,
            const tensor& output,
            const tensor& gradient_input,
            tt::operation_mode mode
        )
        {
            DLIB_CASSERT(have_same_dimensions(grad, output));
            DLIB_CASSERT(have_same_dimensions(grad, gradient_input));
            ttimpl::softmax_gradient(grad.nr() * grad.nc(), grad.k(), grad, output, gradient_input, mode);
        }

        // -----------------------------------------------------------------------------------

        void reorg2(
            bool add_to,
            tensor& dest,
            const int row_stride,
            const int col_stride,
            const tensor& src
        )
        {
            DLIB_CASSERT(!is_same_object(dest, src), "Destination and source must be distinct objects.");
            DLIB_CASSERT(src.nr() % row_stride == 0, "The number of rows in src must be divisible by row_stride.");
            DLIB_CASSERT(src.nc() % col_stride == 0, "The number of columns in src must be divisible by col_stride.");
            DLIB_CASSERT(dest.num_samples() == src.num_samples(), "The number of samples must match.");
            DLIB_CASSERT(dest.k() == src.k() * row_stride * col_stride, "The number of channels must match.");
            DLIB_CASSERT(dest.nr() == src.nr() / row_stride, "The number of rows must match.");
            DLIB_CASSERT(dest.nc() == src.nc() / col_stride, "The number of columns must match.");

            const float* s = src.host();
            float* d = dest.host();

            const size_t sk = src.k(), snr = src.nr(), snc = src.nc();
            const size_t dk = dest.k(), dnr = dest.nr(), dnc = dest.nc(), dsize = dest.size();

            dlib::parallel_for(0, dsize, [&](long i)
                {
                    const size_t out_plane_size = dnr * dnc;
                    const size_t out_sample_size = dk * out_plane_size;

                    const size_t n = i / out_sample_size;
                    const size_t out_idx = i % out_sample_size;
                    const size_t out_k = out_idx / out_plane_size;
                    const size_t out_rc = out_idx % out_plane_size;
                    const size_t out_r = out_rc / dnc;
                    const size_t out_c = out_rc % dnc;

                    const size_t in_k = out_k % sk;
                    const size_t in_r = out_r * row_stride + (out_k / sk) / col_stride;
                    const size_t in_c = out_c * col_stride + (out_k / sk) % col_stride;

                    const size_t in_idx = ((n * sk + in_k) * snr + in_r) * snc + in_c;

                    if (add_to) d[i] += s[in_idx];
                    else d[i] = s[in_idx];
                });
        }

        void reorg_gradient2(
            bool add_to,
            tensor& grad,
            const int row_stride,
            const int col_stride,
            const tensor& gradient_input
        )
        {
            DLIB_CASSERT(!is_same_object(grad, gradient_input), "Grad and gradient_input must be distinct objects.");
            DLIB_CASSERT(grad.nr() % row_stride == 0, "The number of rows in grad must be divisible by row_stride.");
            DLIB_CASSERT(grad.nc() % col_stride == 0, "The number of columns in grad must be divisible by col_stride.");
            DLIB_CASSERT(grad.num_samples() == gradient_input.num_samples(), "The number of samples in grad and gradient_input must match.");
            DLIB_CASSERT(grad.k() == gradient_input.k() / row_stride / col_stride, "The number of channels in grad must be gradient_input.k() divided by row_stride and col_stride.");
            DLIB_CASSERT(grad.nr() == gradient_input.nr() * row_stride, "The number of rows in grad must be gradient_input.nr() multiplied by row_stride.");
            DLIB_CASSERT(grad.nc() == gradient_input.nc() * col_stride, "The number of columns in grad must be gradient_input.nc() multiplied by col_stride.");

            const float* gi = gradient_input.host();
            float* g = grad.host();

            parallel_for(0, gradient_input.num_samples(), [&](long n)
                {
                    for (long k = 0; k < gradient_input.k(); ++k)
                    {
                        for (long r = 0; r < gradient_input.nr(); ++r)
                        {
                            for (long c = 0; c < gradient_input.nc(); ++c)
                            {
                                const auto in_idx = tensor_index(gradient_input, n, k, r, c);
                                const auto out_idx = tensor_index(grad,
                                    n,
                                    k % grad.k(),
                                    r * row_stride + (k / grad.k()) / col_stride,
                                    c * col_stride + (k / grad.k()) % col_stride);

                                if (add_to) g[out_idx] += gi[in_idx];
                                else g[out_idx] = gi[in_idx];
                            }
                        }
                    }
                });
        }

        // -----------------------------------------------------------------------------------

        void embeddings(
            resizable_tensor& dest,
            const tensor& src,
            const tensor& embs
        )
        {
            DLIB_CASSERT(
                src.nr() > 0 &&
                embs.num_samples() > 0 &&
                embs.k() > 0 &&
                embs.nr() == 1 &&
                embs.nc() == 1,
                "\nsrc.num_samples(): " << src.num_samples() <<
                "\nsrc.k(): " << src.k() <<
                "\nsrc.nr(): " << src.nr() <<
                "\nsrc.nc(): " << src.nc() <<
                "\nembs.num_samples(): " << embs.num_samples() <<
                "\nembs.k(): " << embs.k() <<
                "\nembs.nr(): " << embs.nr() <<
                "\nembs.nc(): " << embs.nc()
            );

            long ns = dest.num_samples(), nk = dest.k(), nr = dest.nr(), nc = dest.nc();
            const float* src_data = src.host();
            float* dest_data = dest.host();
            const float* embs_data = embs.host();
            for (long s = 0; s < ns; ++s)
            {
                for (long k = 0; k < nk; ++k)
                {
                    for (long r = 0; r < nr; ++r)
                    {
                        const unsigned long token_idx = static_cast<unsigned long>(src_data[tensor_index(src, s, k, r, 0)]);
                        if (token_idx < embs.num_samples())
                        {
                            for (long c = 0; c < nc; ++c)
                                dest_data[tensor_index(dest, s, k, r, c)] = embs_data[tensor_index(embs, token_idx, c, 0, 0)];
                        }
                        else
                        {
                            for (long c = 0; c < nc; ++c)
                                dest_data[tensor_index(dest, s, k, r, c)] = 0;
                        }
                    }
                }
            }
        }

        void embeddings_gradient(
            const tensor& prev,
            const tensor& gradient_input,
            tensor& grads,
            const tensor& freqs,
            float learning_rate,
            bool scale
        )
        {
            DLIB_CASSERT(
                prev.nr() > 0 &&
                gradient_input.num_samples() == prev.num_samples() &&
                gradient_input.k() == prev.k() &&
                gradient_input.nr() == prev.nr() &&
                gradient_input.nc() == grads.k() &&
                grads.num_samples() > 0 &&
                grads.k() > 0 &&
                grads.nr() == 1 &&
                grads.nc() == 1,
                "\ngradient_input.num_samples(): " << gradient_input.num_samples() <<
                "\ngradient_input.k(): " << gradient_input.k() <<
                "\ngradient_input.nr(): " << gradient_input.nr() <<
                "\ngradient_input.nc(): " << gradient_input.nc() <<
                "\nprev.num_samples(): " << prev.num_samples() <<
                "\nprev.k(): " << prev.k() <<
                "\nprev.nr(): " << prev.nr() <<
                "\nprev.nc(): " << prev.nc() <<
                "\ngrads.num_samples(): " << grads.num_samples() <<
                "\ngrads.k(): " << grads.k() <<
                "\ngrads.nr(): " << grads.nr() <<
                "\ngrads.nc(): " << grads.nc()
            );

            const float* prev_data = prev.host();
            const float* gradient_input_data = gradient_input.host();
            const float* freqs_data = freqs.host();
            float* grads_data = grads.host();
            long ns = gradient_input.num_samples(), nk = gradient_input.k();
            long nr = gradient_input.nr(), nc = gradient_input.nc();

            std::vector<dlib::mutex> embedding_mutexes(grads.num_samples());
            parallel_for(0, ns * nk, [&](long i)
                {
                    long s = i / nk;
                    long k = i % nk;

                    for (long r = 0; r < nr; ++r)
                    {
                        const unsigned long token_idx = static_cast<unsigned long>(prev_data[tensor_index(prev, s, k, r, 0)]);
                        if (token_idx < grads.num_samples())
                        {
                            const float freg_token = freqs_data[token_idx];
                            float freq_scale = 1.0f;

                            if (scale && freg_token != 0.0f) freq_scale = std::min(0.15f, std::max(1.0f / freg_token, 1.0f));
                            auto_mutex locker(embedding_mutexes[token_idx]);
                            for (long c = 0; c < nc; ++c)
                            {
                                const float gradient = gradient_input_data[tensor_index(gradient_input, s, k, r, c)];
                                grads_data[tensor_index(grads, token_idx, c, 0, 0)] -= (gradient * learning_rate * freq_scale);
                            }
                        }
                    }
                });
        }

        // -----------------------------------------------------------------------------------

        void rms_normalize(
            const double eps,
            resizable_tensor& dest,
            resizable_tensor& scale,
            const tensor& src,
            const tensor& gamma
        )
        {
            DLIB_CASSERT(
                gamma.k() == src.k() &&
                gamma.nr() == 1 &&
                gamma.nc() == 1 &&
                eps > 0,
                "\nsrc.k():    " << src.k() <<
                "\ngamma.k():  " << gamma.k() <<
                "\ngamma.nr(): " << gamma.nr() <<
                "\ngamma.nc(): " << gamma.nc() <<
                "\neps:  " << eps
            );

            const long ns = src.num_samples();
            const long ks = src.k();
            const long num = src.nr() * src.nc();

            dest.copy_size(src);
            scale.set_size(ns);

            // Compute RMS values
            scale = 0;
            const float* p_src = src.host();
            float* p_scale = scale.host();
            for (long n = 0; n < ns; ++n)
            {
                for (long k = 0; k < ks; ++k)
                {
                    for (long i = 0; i < num; ++i)
                    {
                        p_scale[n] += (*p_src) * (*p_src);
                        ++p_src;
                    }
                }
                p_scale[n] = 1.0f / std::sqrt(p_scale[n] / (ks * num) + static_cast<float>(eps));
            }
            scale.host();

            // Apply RMS normalization
            p_src = src.host();
            float* p_dest = dest.host();
            const float* p_gamma = gamma.host();
            for (long n = 0; n < ns; ++n)
            {
                for (long k = 0; k < ks; ++k)
                {
                    for (long i = 0; i < num; ++i)
                    {
                        *p_dest = (*p_src) * p_scale[n] * p_gamma[k];
                        ++p_src;
                        ++p_dest;
                    }
                }
            }
        }

        void rms_normalize_gradient(
            const tensor& gradient_input,
            const tensor& scale,
            const tensor& src,
            const tensor& gamma,
            tensor& src_grad,
            tensor& gamma_grad,
            resizable_tensor& dscale
        )
        {
            DLIB_CASSERT(src.num_samples() == scale.size());
            DLIB_CASSERT(have_same_dimensions(gamma, gamma_grad));
            DLIB_CASSERT(gamma.k() == src.k());
            DLIB_CASSERT(gamma.nr() == 1);
            DLIB_CASSERT(gamma.nc() == 1);
            DLIB_CASSERT(have_same_dimensions(gradient_input, src));
            DLIB_CASSERT(have_same_dimensions(gradient_input, src_grad));

            const long ns = src.num_samples();
            const long ks = src.k();
            const long num = src.nr() * src.nc();

            gamma_grad = 0;
            dscale.copy_size(scale);
            dscale = 0;

            auto p_grad = gradient_input.host();
            auto p_src = src.host();
            const auto p_gamma = gamma.host();
            const auto p_gamma_grad = gamma_grad.host();
            const auto p_scale = scale.host();
            auto p_dscale = dscale.host();

            for (long n = 0; n < ns; ++n)
            {
                const float scale_pow = -0.5f * std::pow(p_scale[n], 3.0f);
                for (long k = 0; k < ks; ++k)
                {
                    for (long i = 0; i < num; ++i)
                    {
                        const float x_hat = *p_src * p_scale[n];
                        p_gamma_grad[k] += (*p_grad) * x_hat;

                        const float dx = *p_grad * p_gamma[k];
                        p_dscale[n] += dx * *p_src * scale_pow;

                        ++p_grad;
                        ++p_src;
                    }
                }
            }

            p_grad = gradient_input.host();
            p_src = src.host();
            auto p_src_grad = src_grad.host();
            const float invnum = 1.0f / (ks * num);
            for (long n = 0; n < ns; ++n)
            {
                for (long k = 0; k < ks; ++k)
                {
                    for (long i = 0; i < num; ++i)
                    {
                        const float dx = *p_grad * p_gamma[k];
                        *p_src_grad += dx * p_scale[n] + p_dscale[n] * 2 * *p_src * invnum;

                        ++p_grad;
                        ++p_src;
                        ++p_src_grad;
                    }
                }
            }
        }

        // -----------------------------------------------------------------------------------

        void transpose(
            bool add,
            tensor& dest,
            const tensor& src
        )
        {
            DLIB_CASSERT(dest.num_samples() == src.num_samples() &&
                dest.k() == src.k() &&
                dest.nr() == src.nc() &&
                dest.nc() == src.nr(),
                "Incompatible tensor dimensions.");

            const float* src_data = src.host();
            float* dest_data = dest.host();

            const long num_samples = src.num_samples();
            const long k_dim = src.k();
            const long src_nr = src.nr();
            const long src_nc = src.nc();
            const long dest_nr = dest.nr();
            const long dest_nc = dest.nc();

            parallel_for(0, num_samples * k_dim, [&](long i) {
                const long n = i / k_dim;
                const long k = i % k_dim;
                const long src_nk_offset = (n * src.k() + k) * src_nr;
                const long dest_nk_offset = (n * dest.k() + k) * dest_nr;

                for (long r = 0; r < src_nr; ++r) {
                    for (long c = 0; c < src_nc; ++c) {
                        const long src_idx = (src_nk_offset + r) * src_nc + c;
                        const long dest_idx = (dest_nk_offset + c) * dest_nc + r;

                        if (add) dest_data[dest_idx] += src_data[src_idx];
                        else dest_data[dest_idx] = src_data[src_idx];
                    }
                }
            });
        }

        // -----------------------------------------------------------------------------------

        void split_columns(
            bool add_to,
            tensor& dest,
            const tensor& src,
            const long num_heads
        ) {
            DLIB_CASSERT(is_same_object(dest, src) == false);
            DLIB_CASSERT(dest.num_samples() == src.num_samples() &&
                dest.k() == num_heads &&
                src.k() == 1 &&
                dest.nc() == (src.nc() / num_heads) &&
                src.nc() % num_heads == 0,
                "Incompatible tensor dimensions.");

            for (long s = 0; s < dest.num_samples(); ++s)
            {
                for (long k = 0; k < dest.k(); ++k)
                {
                    for (long r = 0; r < dest.nr(); ++r)
                    {
                        for (long c = 0; c < dest.nc(); ++c)
                        {
                            if (add_to) dest.host()[tensor_index(dest, s, k, r, c)] += src.host()[tensor_index(src, s, 0, r, (k * dest.nc()) + c)];
                            else dest.host()[tensor_index(dest, s, k, r, c)] = src.host()[tensor_index(src, s, 0, r, (k * dest.nc()) + c)];
                        }
                    }
                }
            }
        }

        // -----------------------------------------------------------------------------------

        void merge_columns(
            bool add_to,
            tensor& dest,
            const tensor& src
        ) {
            DLIB_CASSERT(is_same_object(dest, src) == false);
            DLIB_CASSERT(dest.num_samples() == src.num_samples() &&
                dest.k() == 1 &&
                src.k() > 1 &&
                dest.nr() == src.nr() &&
                dest.nc() == (src.nc() * src.k()),
                "Incompatible tensor dimensions.");

            for (long s = 0; s < src.num_samples(); ++s)
            {
                for (long k = 0; k < src.k(); ++k)
                {
                    for (long r = 0; r < src.nr(); ++r)
                    {
                        for (long c = 0; c < src.nc(); ++c)
                        {
                            if (add_to) dest.host()[tensor_index(dest, s, 0, r, (k * src.nc()) + c)] += src.host()[tensor_index(src, s, k, r, c)];
                            else dest.host()[tensor_index(dest, s, 0, r, (k * src.nc()) + c)] = src.host()[tensor_index(src, s, k, r, c)];
                        }
                    }
                }
            }
        }

        // -----------------------------------------------------------------------------------
    }

#ifdef DLIB_USE_CUDA
    namespace cuda {
        class cublas_context
        {
        public:
            cublas_context(const cublas_context&) = delete;
            cublas_context& operator=(const cublas_context&) = delete;

            cublas_context()
            {
                handles.resize(16);
            }
            ~cublas_context()
            {
                for (auto h : handles)
                {
                    if (h)
                        cublasDestroy(h);
                }
            }

            cublasHandle_t get_handle()
            {
                int new_device_id;
                CHECK_CUDA(cudaGetDevice(&new_device_id));
                if (new_device_id >= (long)handles.size()) handles.resize(new_device_id + 16);
                if (!handles[new_device_id]) cublasCreate(&handles[new_device_id]);
                return handles[new_device_id];
            }

        private:
            std::vector<cublasHandle_t> handles;
        };

        static cublasHandle_t context()
        {
            thread_local cublas_context c;
            return c.get_handle();
        }

        static cudnnTensorDescriptor_t descriptor(const tensor& t)
        {
            return (const cudnnTensorDescriptor_t)t.get_cudnn_tensor_descriptor().get_handle();
        }
        static cudnnTensorDescriptor_t descriptor(const tensor_descriptor& t)
        {
            return (const cudnnTensorDescriptor_t)t.get_handle();
        }

        class cudnn_context
        {
        public:
            // not copyable
            cudnn_context(const cudnn_context&) = delete;
            cudnn_context& operator=(const cudnn_context&) = delete;

            cudnn_context()
            {
                handles.resize(16);
            }
            ~cudnn_context()
            {
                for (auto h : handles)
                {
                    if (h)
                        cudnnDestroy(h);
                }
            }

            cudnnHandle_t get_handle(
            )
            {
                int new_device_id;
                CHECK_CUDA(cudaGetDevice(&new_device_id));
                // make room for more devices if needed
                if (new_device_id >= (long)handles.size())
                    handles.resize(new_device_id + 16);

                // If we don't have a handle already for this device then make one
                if (!handles[new_device_id])
                    CHECK_CUDNN(cudnnCreate(&handles[new_device_id]));

                // Finally, return the handle for the current device
                return handles[new_device_id];
            }

        private:

            std::vector<cudnnHandle_t> handles;
        };

        static cudnnHandle_t ccontext()
        {
            thread_local cudnn_context c;
            return c.get_handle();
        }

        void gemm(
            float beta,
            tensor& dest,
            float alpha,
            const tensor& lhs,
            bool trans_lhs,
            const tensor& rhs,
            bool trans_rhs,
            tt::operation_mode mode = tt::operation_mode::CHANNEL_WISE
        )
        {
            if (mode == tt::operation_mode::CHANNEL_WISE)
            {
                // Recall that BLAS uses column major order so to deal with that we flip the
                // order of the lhs and rhs arguments.
                const auto transa = trans_lhs ? CUBLAS_OP_T : CUBLAS_OP_N;
                const auto transb = trans_rhs ? CUBLAS_OP_T : CUBLAS_OP_N;

                const int dest_nr = dest.num_samples();
                const int dest_nc = dest.size() / dest_nr;
                const int lhs_nr = lhs.num_samples();
                const int lhs_nc = lhs.size() / lhs_nr;
                const int rhs_nr = rhs.num_samples();
                const int rhs_nc = rhs.size() / rhs_nr;
                if (trans_lhs && trans_rhs)
                {
                    DLIB_ASSERT(dest_nr == lhs_nc &&
                        dest_nc == rhs_nr &&
                        lhs_nr == rhs_nc)
                }
                else if (!trans_lhs && trans_rhs)
                {
                    DLIB_ASSERT(dest_nr == lhs_nr &&
                        dest_nc == rhs_nr &&
                        lhs_nc == rhs_nc)
                }
                else if (trans_lhs && !trans_rhs)
                {
                    DLIB_ASSERT(dest_nr == lhs_nc &&
                        dest_nc == rhs_nc &&
                        lhs_nr == rhs_nr)
                }
                else
                {
                    DLIB_ASSERT(dest_nr == lhs_nr &&
                        dest_nc == rhs_nc &&
                        lhs_nc == rhs_nr)
                }

                const int k = trans_rhs ? rhs_nc : rhs_nr;
                CHECK_CUBLAS(cublasSgemm(context(),
                    transb,
                    transa,
                    dest_nc, dest_nr, k,
                    &alpha,
                    rhs.device(), rhs_nc,
                    lhs.device(), lhs_nc,
                    &beta,
                    dest.device(), dest_nc));
            }
            else if (mode == tt::operation_mode::PLANE_WISE)
            {
                const auto transa = trans_lhs ? CUBLAS_OP_T : CUBLAS_OP_N;
                const auto transb = trans_rhs ? CUBLAS_OP_T : CUBLAS_OP_N;

                long num_samples = std::min({ lhs.num_samples(), rhs.num_samples(), dest.num_samples() });
                long num_channels = std::min({ lhs.k(), rhs.k(), dest.k() });

                auto is_matrix = [](const auto& tensor) {
                    return ((tensor.num_samples() * tensor.k() == 1 && tensor.nr() * tensor.nc() > 1) ||
                        (tensor.num_samples() * tensor.k() > 1 && tensor.nr() * tensor.nc() == 1));
                    };
                const bool lhs_is_matrix = is_matrix(lhs), rhs_is_matrix = is_matrix(rhs), dest_is_matrix = is_matrix(dest);
                if (lhs_is_matrix && rhs_is_matrix && dest_is_matrix) num_samples = num_channels = 1;

                size_t lhs_rows = lhs.nr();
                size_t lhs_cols = lhs.nc();
                if (lhs_is_matrix && (lhs.num_samples() > 1 || lhs.k() > 1)) {
                    lhs_rows = lhs.num_samples();
                    lhs_cols = lhs.k();
                }
                size_t rhs_rows = rhs.nr();
                size_t rhs_cols = rhs.nc();
                if (rhs_is_matrix && (rhs.num_samples() > 1 || rhs.k() > 1)) {
                    rhs_rows = rhs.num_samples();
                    rhs_cols = rhs.k();
                }
                size_t dest_rows = dest.nr();
                size_t dest_cols = dest.nc();
                if (dest_is_matrix && (dest.num_samples() > 1 || dest.k() > 1)) {
                    dest_rows = dest.num_samples();
                    dest_cols = dest.k();
                }

                const size_t lhs_plane_size = lhs_rows * lhs_cols;
                const size_t rhs_plane_size = rhs_rows * rhs_cols;
                const size_t dest_plane_size = dest_rows * dest_cols;

                for (size_t b = 0; b < num_samples; ++b)
                {
                    for (size_t c = 0; c < num_channels; ++c)
                    {
                        auto lhs_slice = lhs_is_matrix ? lhs.device() :
                            lhs.device() + (b * num_channels + c) * lhs_plane_size;
                        auto rhs_slice = rhs_is_matrix ? rhs.device() :
                            rhs.device() + (b * num_channels + c) * rhs_plane_size;
                        auto dest_slice = dest_is_matrix ? dest.device() :
                            dest.device() + (b * num_channels + c) * dest_plane_size;
                        const int k = trans_rhs ? rhs_cols : rhs_rows;

                        CHECK_CUBLAS(cublasSgemm(
                            context(), transb, transa, dest_cols, dest_rows, k,
                            &alpha, rhs_slice, rhs_cols, lhs_slice, lhs_cols,
                            &beta, dest_slice, dest_cols
                        ));
                    }
                }
            }
        }

        void softmax(
            tensor& dest,
            const tensor& src,
            tt::operation_mode mode = tt::operation_mode::CHANNEL_WISE
        )
        {
            DLIB_CASSERT(have_same_dimensions(dest, src));
            DLIB_CASSERT(mode == tt::operation_mode::CHANNEL_WISE ||
                mode == tt::operation_mode::PLANE_WISE, "Invalid softmax mode");
            if (src.size() == 0) return;

            const float alpha = 1;
            const float beta = 0;

            if (mode == tt::operation_mode::CHANNEL_WISE)
            {
                CHECK_CUDNN(cudnnSoftmaxForward(ccontext(),
                    CUDNN_SOFTMAX_ACCURATE,
                    CUDNN_SOFTMAX_MODE_CHANNEL,
                    &alpha,
                    descriptor(src),
                    src.device(),
                    &beta,
                    descriptor(dest),
                    dest.device()));
            }
            else if (mode == tt::operation_mode::PLANE_WISE)
            {
                const size_t num_samples = src.num_samples();
                const size_t num_channels = src.k();
                const size_t plane_size = src.nr() * src.nc();

                for (size_t s = 0; s < num_samples; ++s)
                {
                    for (size_t k = 0; k < num_channels; ++k)
                    {
                        auto src_slice = src.device() + (s * num_channels + k) * plane_size;
                        auto dest_slice = dest.device() + (s * num_channels + k) * plane_size;
                        auto a_src_slice = alias_tensor(src.nr(), src.nc())(src, (s * num_channels + k) * plane_size);
                        auto a_dest_slice = alias_tensor(dest.nr(), dest.nc())(dest, (s * num_channels + k) * plane_size);

                        CHECK_CUDNN(cudnnSoftmaxForward(ccontext(),
                            CUDNN_SOFTMAX_ACCURATE,
                            CUDNN_SOFTMAX_MODE_CHANNEL,
                            &alpha,
                            descriptor(a_src_slice),
                            src_slice,
                            &beta,
                            descriptor(a_dest_slice),
                            dest_slice));
                    }
                }
            }
        }

        void softmax_gradient(
            tensor& grad,
            const tensor& output,
            const tensor& gradient_input,
            tt::operation_mode mode = tt::operation_mode::CHANNEL_WISE
        )
        {
            DLIB_CASSERT(
                have_same_dimensions(output, gradient_input) == true &&
                have_same_dimensions(output, grad) == true);
            DLIB_CASSERT(mode == tt::operation_mode::CHANNEL_WISE ||
                mode == tt::operation_mode::PLANE_WISE, "Invalid softmax mode");
            if (output.size() == 0) return;

            const float alpha = 1;
            const float beta = is_same_object(grad, gradient_input) ? 0 : 1;

            if (mode == tt::operation_mode::CHANNEL_WISE)
            {
                CHECK_CUDNN(cudnnSoftmaxBackward(ccontext(),
                    CUDNN_SOFTMAX_ACCURATE,
                    CUDNN_SOFTMAX_MODE_CHANNEL,
                    &alpha,
                    descriptor(output),
                    output.device(),
                    descriptor(gradient_input),
                    gradient_input.device(),
                    &beta,
                    descriptor(grad),
                    grad.device()));
            }
            else if (mode == tt::operation_mode::PLANE_WISE)
            {
                const size_t num_samples = output.num_samples();
                const size_t num_channels = output.k();
                const size_t plane_size = output.nr() * output.nc();

                for (size_t s = 0; s < num_samples; ++s)
                {
                    for (size_t k = 0; k < num_channels; ++k)
                    {
                        auto output_slice = output.device() + (s * num_channels + k) * plane_size;
                        auto gi_slice = gradient_input.device() + (s * num_channels + k) * plane_size;
                        auto grad_slice = grad.device() + (s * num_channels + k) * plane_size;
                        auto a_output_slice = alias_tensor(output.nr(), output.nc())(output, (s * num_channels + k) * plane_size);
                        auto a_gi_slice = alias_tensor(gradient_input.nr(), gradient_input.nc())(gradient_input, (s * num_channels + k) * plane_size);
                        auto a_grad_slice = alias_tensor(grad.nr(), grad.nc())(grad, (s * num_channels + k) * plane_size);

                        CHECK_CUDNN(cudnnSoftmaxBackward(ccontext(),
                            CUDNN_SOFTMAX_ACCURATE,
                            CUDNN_SOFTMAX_MODE_CHANNEL,
                            &alpha,
                            descriptor(a_output_slice),
                            output_slice,
                            descriptor(a_gi_slice),
                            gi_slice,
                            &beta,
                            descriptor(a_grad_slice),
                            grad_slice));
                    }
                }
            }
        }
    }
#endif

    namespace tt {
        /* TO BE ADDED TO <tensor_tools.cpp> */
        // ----------------------------------------------------------------------------------------

        void embeddings(
            resizable_tensor& dest,
            const tensor& src,
            const tensor& embs
        )
        {
#ifdef DLIB_USE_CUDA
            cuda::embeddings(dest, src, embs);
#else
            cpu::embeddings(dest, src, embs);
#endif
        }

        void embeddings_gradient(
            const tensor& prev,
            const tensor& gradient_input,
            tensor& grads,
            const tensor& freqs,
            float learning_rate,
            bool scale
        )
        {
#ifdef DLIB_USE_CUDA
            cuda::embeddings_gradient(prev, gradient_input, grads, freqs, learning_rate, scale);
#else
            cpu::embeddings_gradient(prev, gradient_input, grads, freqs, learning_rate, scale);
#endif
        }

        void softmax(
            tensor& dest,
            const tensor& src,
            tt::operation_mode mode = tt::operation_mode::CHANNEL_WISE
        )
        {
#ifdef DLIB_USE_CUDA
            cuda::softmax(dest, src, mode);
#else
            cpu::softmax(dest, src, mode);
#endif
        }

        void softmax_gradient(
            tensor& grad,
            const tensor& dest,
            const tensor& gradient_input,
            tt::operation_mode mode = tt::operation_mode::CHANNEL_WISE
        )
        {
#ifdef DLIB_USE_CUDA
            cuda::softmax_gradient(grad, dest, gradient_input, mode);
#else
            cpu::softmax_gradient(grad, dest, gradient_input, mode);
#endif
        }

        void gemm(
            float beta,
            tensor& dest,
            float alpha,
            const tensor& lhs,
            bool trans_lhs,
            const tensor& rhs,
            bool trans_rhs,
            tt::operation_mode mode = tt::operation_mode::CHANNEL_WISE
        )
        {
#ifdef DLIB_USE_CUDA
            cuda::gemm(beta, dest, alpha, lhs, trans_lhs, rhs, trans_rhs, mode);
#else
            if (mode == tt::operation_mode::CHANNEL_WISE)
            {
                if (beta != 0)
                {
                    if (trans_lhs && trans_rhs)
                        dest = alpha * trans(mat(lhs)) * trans(mat(rhs)) + beta * mat(dest);
                    else if (!trans_lhs && trans_rhs)
                        dest = alpha * mat(lhs) * trans(mat(rhs)) + beta * mat(dest);
                    else if (trans_lhs && !trans_rhs)
                        dest = alpha * trans(mat(lhs)) * mat(rhs) + beta * mat(dest);
                    else
                        dest = alpha * mat(lhs) * mat(rhs) + beta * mat(dest);
                }
                else
                {
                    if (trans_lhs && trans_rhs)
                        dest = alpha * trans(mat(lhs)) * trans(mat(rhs));
                    else if (!trans_lhs && trans_rhs)
                        dest = alpha * mat(lhs) * trans(mat(rhs));
                    else if (trans_lhs && !trans_rhs)
                        dest = alpha * trans(mat(lhs)) * mat(rhs);
                    else
                        dest = alpha * mat(lhs) * mat(rhs);
                }
            }
            else if (mode == tt::operation_mode::PLANE_WISE)
            {
                auto is_matrix = [](const auto& tensor) {
                    return ((tensor.num_samples() * tensor.k() == 1 && tensor.nr() * tensor.nc() > 1) ||
                        (tensor.num_samples() * tensor.k() > 1 && tensor.nr() * tensor.nc() == 1));
                    };

                long num_samples = std::min({ lhs.num_samples(), rhs.num_samples(), dest.num_samples() });
                long num_channels = std::min({ lhs.k(), rhs.k(), dest.k() });
                const bool lhs_is_matrix = is_matrix(lhs), rhs_is_matrix = is_matrix(rhs), dest_is_matrix = is_matrix(dest);

                if (lhs_is_matrix && rhs_is_matrix && dest_is_matrix) {
                    num_samples = num_channels = 1;
                }                

                size_t lhs_rows = (lhs_is_matrix && lhs.num_samples() > 1) ? lhs.num_samples() : lhs.nr();
                size_t lhs_cols = (lhs_is_matrix && lhs.k() > 1) ? lhs.k() : lhs.nc();
                size_t rhs_rows = (rhs_is_matrix && rhs.num_samples() > 1) ? rhs.num_samples() : rhs.nr();
                size_t rhs_cols = (rhs_is_matrix && rhs.k() > 1) ? rhs.k() : rhs.nc();
                size_t dest_rows = (dest_is_matrix && dest.num_samples() > 1) ? dest.num_samples() : dest.nr();
                size_t dest_cols = (dest_is_matrix && dest.k() > 1) ? dest.k() : dest.nc();

                const size_t lhs_plane_size = lhs_rows * lhs_cols;
                const size_t rhs_plane_size = rhs_rows * rhs_cols;
                const size_t dest_plane_size = dest_rows * dest_cols;

                for (size_t b = 0; b < num_samples; ++b)
                {
                    for (size_t c = 0; c < num_channels; ++c)
                    {
                        auto lhs_slice = lhs_is_matrix ? alias_tensor(lhs_rows, lhs_cols)(lhs, 0) :
                            alias_tensor(lhs_rows, lhs_cols)(lhs, (b * num_channels + c) * lhs_plane_size);
                        auto rhs_slice = rhs_is_matrix ? alias_tensor(rhs_rows, rhs_cols)(rhs, 0) :
                            alias_tensor(rhs_rows, rhs_cols)(rhs, (b * num_channels + c) * rhs_plane_size);
                        auto dest_slice = dest_is_matrix ? alias_tensor(dest_rows, dest_cols)(dest, 0) :
                            alias_tensor(dest_rows, dest_cols)(dest, (b * num_channels + c) * dest_plane_size);

                        if (beta != 0)
                        {
                            if (trans_lhs && trans_rhs)
                                dest_slice = alpha * trans(mat(lhs_slice)) * trans(mat(rhs_slice)) + beta * mat(dest_slice);
                            else if (!trans_lhs && trans_rhs)
                                dest_slice = alpha * mat(lhs_slice) * trans(mat(rhs_slice)) + beta * mat(dest_slice);
                            else if (trans_lhs && !trans_rhs)
                                dest_slice = alpha * trans(mat(lhs_slice)) * mat(rhs_slice) + beta * mat(dest_slice);
                            else
                                dest_slice = alpha * mat(lhs_slice) * mat(rhs_slice) + beta * mat(dest_slice);
                        }
                        else
                        {
                            if (trans_lhs && trans_rhs)
                                dest_slice = alpha * trans(mat(lhs_slice)) * trans(mat(rhs_slice));
                            else if (!trans_lhs && trans_rhs)
                                dest_slice = alpha * mat(lhs_slice) * trans(mat(rhs_slice));
                            else if (trans_lhs && !trans_rhs)
                                dest_slice = alpha * trans(mat(lhs_slice)) * mat(rhs_slice);
                            else
                                dest_slice = alpha * mat(lhs_slice) * mat(rhs_slice);
                        }
                    }
                }
            }
#endif
        }

        // ----------------------------------------------------------------------------------------

        void rms_normalize(
            const double eps,
            resizable_tensor& dest,
            resizable_tensor& scale,
            const tensor& src,
            const tensor& gamma
        )
        {
#ifdef DLIB_USE_CUDA
            cuda::rms_normalize(eps, dest, scale, src, gamma);
#else
            cpu::rms_normalize(eps, dest, scale, src, gamma);
#endif
        }

        void rms_normalize_gradient(
            const tensor& gradient_input,
            const tensor& scale,
            const tensor& src,
            const tensor& gamma,
            tensor& src_grad,
            tensor& gamma_grad,
            resizable_tensor& dscale
        )
        {
#ifdef DLIB_USE_CUDA
            cuda::rms_normalize_gradient(gradient_input, scale, src, gamma, src_grad, gamma_grad, dscale);
#else
            cpu::rms_normalize_gradient(gradient_input, scale, src, gamma, src_grad, gamma_grad, dscale);
#endif
        }

        // ----------------------------------------------------------------------------------------

        void reorg2(
            bool add_to,
            tensor& dest,
            const int row_stride,
            const int col_stride,
            const tensor& src
        )
        {
#ifdef DLIB_USE_CUDA
            cuda::reorg2(add_to, dest, row_stride, col_stride, src);
#else
            cpu::reorg2(add_to, dest, row_stride, col_stride, src);
#endif
        }

        void reorg_gradient2(
            bool add_to,
            tensor& grad,
            const int row_stride,
            const int col_stride,
            const tensor& gradient_input
        )
        {
#ifdef DLIB_USE_CUDA
            cuda::reorg_gradient2(add_to, grad, row_stride, col_stride, gradient_input);
#else
            cpu::reorg_gradient2(add_to, grad, row_stride, col_stride, gradient_input);
#endif
        }

        // ----------------------------------------------------------------------------------------

        void transpose(
            bool add_to,
            tensor& dest,
            const tensor& src
        )
        {
#ifdef DLIB_USE_CUDA
            cuda::transpose(add_to, dest, src);
#else
            cpu::transpose(add_to, dest, src);
#endif
        }

        // ----------------------------------------------------------------------------------------
        void split_columns(
            bool add_to,
            tensor& dest,
            const tensor& src,
            const long num_heads
        )
        {
#ifdef DLIB_USE_CUDA
            cuda::split_columns(add_to, dest, src, num_heads);
#else
            cpu::split_columns(add_to, dest, src, num_heads);
#endif
        }

        // ----------------------------------------------------------------------------------------
        void merge_columns(
            bool add_to,
            tensor& dest,
            const tensor& src
        )
        {
#ifdef DLIB_USE_CUDA
            cuda::merge_columns(add_to, dest, src);
#else
            cpu::merge_columns(add_to, dest, src);
#endif
        }
    }

    /* TO BE ADDED TO <layers.h> */
    // ----------------------------------------------------------------------------------------

    const double DEFAULT_RMS_NORM_EPS = 1e-5;

    class rms_norm_
    {
    public:
        explicit rms_norm_(
            double eps_ = DEFAULT_RMS_NORM_EPS
        ) :
            learning_rate_multiplier(1),
            weight_decay_multiplier(0),
            bias_learning_rate_multiplier(1),
            bias_weight_decay_multiplier(1),
            eps(eps_)
        {
        }

        double get_eps() const { return eps; }

        double get_learning_rate_multiplier() const { return learning_rate_multiplier; }
        double get_weight_decay_multiplier() const { return weight_decay_multiplier; }
        void set_learning_rate_multiplier(double val) { learning_rate_multiplier = val; }
        void set_weight_decay_multiplier(double val) { weight_decay_multiplier = val; }

        double get_bias_learning_rate_multiplier() const { return bias_learning_rate_multiplier; }
        double get_bias_weight_decay_multiplier() const { return bias_weight_decay_multiplier; }
        void set_bias_learning_rate_multiplier(double val) { bias_learning_rate_multiplier = val; }
        void set_bias_weight_decay_multiplier(double val) { bias_weight_decay_multiplier = val; }

        inline dpoint map_input_to_output(const dpoint& p) const { return p; }
        inline dpoint map_output_to_input(const dpoint& p) const { return p; }

        template <typename SUBNET>
        void setup(const SUBNET& sub)
        {
            gamma = alias_tensor(1, sub.get_output().k());
            params.set_size(gamma.size());
            gamma(params, 0) = 1;
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            auto g = gamma(params, 0);
            tt::rms_normalize(eps, output, scale, sub.get_output(), g);
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& params_grad)
        {
            auto g = gamma(params, 0);
            auto g_grad = gamma(params_grad, 0);
            tt::rms_normalize_gradient(gradient_input, scale, sub.get_output(), g, sub.get_gradient_input(), g_grad, dscale);
        }

        const tensor& get_layer_params() const { return params; };
        tensor& get_layer_params() { return params; };

        friend void serialize(const rms_norm_& item, std::ostream& out)
        {
            serialize("rms_norm_", out);
            serialize(item.params, out);
            serialize(item.gamma, out);
            serialize(item.learning_rate_multiplier, out);
            serialize(item.weight_decay_multiplier, out);
            serialize(item.bias_learning_rate_multiplier, out);
            serialize(item.bias_weight_decay_multiplier, out);
            serialize(item.eps, out);
        }

        friend void deserialize(rms_norm_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "rms_norm_")
                throw serialization_error("Unexpected version '" + version + "' found while deserializing dlib::rms_norm_.");
            deserialize(item.params, in);
            deserialize(item.gamma, in);
            deserialize(item.learning_rate_multiplier, in);
            deserialize(item.weight_decay_multiplier, in);
            deserialize(item.bias_learning_rate_multiplier, in);
            deserialize(item.bias_weight_decay_multiplier, in);
            deserialize(item.eps, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const rms_norm_& item)
        {
            out << "rms_norm";
            out << " (eps=" << item.eps << ")";
            out << " learning_rate_mult=" << item.learning_rate_multiplier;
            out << " weight_decay_mult=" << item.weight_decay_multiplier;
            out << " bias_learning_rate_mult=" << item.bias_learning_rate_multiplier;
            out << " bias_weight_decay_mult=" << item.bias_weight_decay_multiplier;
            return out;
        }

        friend void to_xml(const rms_norm_& item, std::ostream& out)
        {
            out << "<rms_norm";
            out << " eps='" << item.eps << "'";
            out << " learning_rate_mult='" << item.learning_rate_multiplier << "'";
            out << " weight_decay_mult='" << item.weight_decay_multiplier << "'";
            out << " bias_learning_rate_mult='" << item.bias_learning_rate_multiplier << "'";
            out << " bias_weight_decay_mult='" << item.bias_weight_decay_multiplier << "'";
            out << ">\n";
            out << mat(item.params);
            out << "</rms_norm>\n";
        }

    private:
        resizable_tensor params;
        alias_tensor gamma;
        resizable_tensor scale;
        resizable_tensor dscale;
        double learning_rate_multiplier;
        double weight_decay_multiplier;
        double bias_learning_rate_multiplier;
        double bias_weight_decay_multiplier;
        double eps;
    };

    template <typename SUBNET>
    using rms_norm = add_layer<rms_norm_, SUBNET>;

    // ----------------------------------------------------------------------------------------

    class display_tensor_ {
    public:
        display_tensor_() {}
        template <typename SUBNET> void setup(const SUBNET& /* sub */) {}

        template <typename SUBNET> void forward(const SUBNET& sub, resizable_tensor& output) {
            auto& prev = sub.get_output();
            output.copy_size(prev);
            tt::copy_tensor(false, output, 0, prev, 0, prev.k());
            DBG_INFO("display_tensor.forward: ", output, false);
        }
        template <typename SUBNET> void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/) {
            auto& prev = sub.get_gradient_input();
            tt::copy_tensor(true, prev, 0, gradient_input, 0, gradient_input.k());
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const display_tensor_& /* item */, std::ostream& out) {}
        friend void deserialize(display_tensor_& /* item */, std::istream& in) {}

        friend std::ostream& operator<<(std::ostream& out, const display_tensor_& /* item */) {
            out << "display_tensor";
            return out;
        }
        friend void to_xml(const display_tensor_& /* item */, std::ostream& out) {
            out << "<display_tensor />\n";
        }
    private:
        dlib::resizable_tensor params; // unused
    };
    template <typename SUBNET> using display_tensor = add_layer<display_tensor_, SUBNET>;

    // ----------------------------------------------------------------------------------------
    /* TO BE ADDED TO <layers_abstract.h> & <layers.h> */

    template <int DROP_RATE_PERCENT>
    class dropout_rate_ : public dropout_
    {
    public:
        explicit dropout_rate_() : dropout_(static_cast<float>(DROP_RATE_PERCENT) / 100.0f)
        {
            static_assert(DROP_RATE_PERCENT >= 0 && DROP_RATE_PERCENT <= 100,
                "DROP_RATE_PERCENT must be between 0 and 100, inclusive.");
        }
    };
    template <int DROP_RATE, typename SUBNET>
    using dropout_rate = add_layer<dropout_rate_<DROP_RATE>, SUBNET>;

    // ----------------------------------------------------------------------------------------
    /* TO BE ADDED TO <layers.h> */

    template<
        unsigned long num_embeddings_,
        unsigned long embedding_dim_
    >
    class embeddings_
    {
        static_assert(num_embeddings_ > 0, "The size of the embedding dictionary must be > 0");
        static_assert(embedding_dim_ > 0, "The size of each embedding vector must be > 0");

    public:
        embeddings_() : num_embeddings(num_embeddings_),
            embedding_dim(embedding_dim_),
            learning_rate_multiplier(1.0f),
            scale_by_freq(true)
        {
        }

        double get_learning_rate_multiplier() const { return learning_rate_multiplier; }
        void set_learning_rate_multiplier(double val) { learning_rate_multiplier = val; }

        void set_scale_by_freq(bool val) { scale_by_freq = val; }
        bool get_scale_by_freq() const { return scale_by_freq; }

        unsigned long get_num_embeddings() const { return num_embeddings; }
        void set_num_embeddings(unsigned long num)
        {
            DLIB_CASSERT(num > 0);
            if (num != num_embeddings)
            {
                DLIB_CASSERT(get_embeddings().size() == 0,
                    "It is not possible to change the size of the embedding dictionary if the parameter has already been assigned.");
            }
        }

        unsigned long get_embedding_dim() const { return embedding_dim; }
        void set_embedding_dim(unsigned long dim)
        {
            DLIB_CASSERT(dim > 0);
            if (dim != embedding_dim)
            {
                DLIB_CASSERT(get_embeddings().size() == 0,
                    "It is not possible to change the size of the embedding dictionary if the parameter has already been assigned.");
            }
        }

        template <typename SUBNET>
        void setup(const SUBNET& /*sub*/)
        {
            embs.set_size(num_embeddings, embedding_dim);
            tt::tensor_rand rnd(std::rand());
            rnd.fill_gaussian(embs);
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            const auto& prev = sub.get_output();
            output.set_size(prev.num_samples(), prev.k(), prev.nr(), embedding_dim);

            tt::embeddings(output, prev, embs);
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            // Since this class is expected to be directly after an <input> layer,
            // it's not necessary to propagate the gradient.
            // Additionally, this layer is treated as constant during backpropagation,
            // so it technically doesn't contribute to the gradient computation.
            if (learning_rate_multiplier != 0)
            {
                auto& prev_src = sub.get_output();

                calc_token_freqs(prev_src, gradient_input);
                tt::embeddings_gradient(prev_src, gradient_input, embs, freqs, learning_rate_multiplier, scale_by_freq);
            }
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        const tensor& get_embeddings() const { return embs; }
        tensor& get_embeddings() { return embs; }

        friend void serialize(const embeddings_& item, std::ostream& out)
        {
            serialize("embeddings_", out);
            serialize(item.embs, out);
            serialize(item.num_embeddings, out);
            serialize(item.embedding_dim, out);
            serialize(item.learning_rate_multiplier, out);
            serialize(item.scale_by_freq, out);
        }
        friend void deserialize(embeddings_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "embeddings_")
                throw serialization_error("Unexpected version found while deserializing dlib::embeddings_.");
            deserialize(item.embs, in);
            deserialize(item.num_embeddings, in);
            deserialize(item.embedding_dim, in);
            deserialize(item.learning_rate_multiplier, in);
            deserialize(item.scale_by_freq, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const embeddings_& item)
        {
            out << "embeddings (num_embeddings=" << item.num_embeddings
                << ", embedding_dim=" << item.embedding_dim
                << ") learning_rate_mult=" << item.learning_rate_multiplier;
            return out;
        }
        friend void to_xml(const embeddings_& item, std::ostream& out)
        {
            out << "<embeddings num_embeddings='" << item.num_embeddings
                << "' embedding_dim='" << item.embedding_dim
                << "' learning_rate_mult='"
                << item.learning_rate_multiplier << "'>\n";
            out << mat(item.embs);
            out << "</embeddings>\n";
        }

    private:
        void calc_token_freqs(const tensor& prev, const tensor& input) {
            if (freqs.size() == 0) freqs.set_size(num_embeddings, 1, 1, 1);
            freqs = 0;

            const float* prev_data = prev.host();
            float* freqs_data = freqs.host();
            for (long s = 0; s < input.num_samples(); ++s)
            {
                for (long k = 0; k < input.k(); ++k)
                {
                    for (long r = 0; r < input.nr(); ++r)
                    {
                        const unsigned long token_idx = static_cast<unsigned long>(prev_data[tensor_index(prev, s, k, r, 0)]);
                        if (token_idx < num_embeddings) freqs_data[tensor_index(freqs, token_idx, 0, 0, 0)]++;
                    }
                }
            }
        }

        resizable_tensor params; // unused
        resizable_tensor embs, freqs;
        unsigned long num_embeddings, embedding_dim;
        double learning_rate_multiplier;
        bool scale_by_freq;
    };

    template <unsigned long nb_embeddings, unsigned long embedding_length, typename SUBNET>
    using embeddings = add_layer<embeddings_<nb_embeddings, embedding_length>, SUBNET>;

    class positional_encodings_ {
    public:
        positional_encodings_(unsigned long sequence_dim_ = 1, unsigned long embedding_dim_ = 1) :
            sequence_dim(sequence_dim_), embedding_dim(embedding_dim_)
        {
        }
        positional_encodings_(const positional_encodings_& item) :
            pe(item.pe), sequence_dim(item.sequence_dim), embedding_dim(item.embedding_dim)
        {
        }
        positional_encodings_& operator= (const positional_encodings_& item)
        {
            if (this == &item) return *this;
            pe = item.pe;
            sequence_dim = item.sequence_dim;
            embedding_dim = item.embedding_dim;
            return *this;
        }

        template <typename SUBNET>
        void setup(const SUBNET& sub)
        {
            auto& prev = sub.get_output();

            sequence_dim = prev.nr();
            embedding_dim = prev.nc();
            const unsigned long ns = prev.num_samples();
            const unsigned long nk = prev.k();
            const float n = 10000.0f;

            pe.set_size(ns, nk, sequence_dim, embedding_dim);
            for (unsigned long s = 0; s < ns; ++s)
            {
                for (unsigned long k = 0; k < nk; ++k)
                {
                    for (unsigned long r = 0; r < sequence_dim; ++r)
                    {
                        for (unsigned long c = 0; c < embedding_dim; ++c)
                        {
                            float theta = static_cast<float>(r) / std::pow(n, static_cast<float>(c) / embedding_dim);
                            if (c % 2 == 0) pe.host()[tensor_index(pe, s, k, r, c)] = std::sin(theta);
                            else pe.host()[tensor_index(pe, s, k, r, c)] = std::cos(theta);
                        }
                    }
                }
            }
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {            
            const auto& prev_output = sub.get_output();
            if (!have_same_dimensions(prev_output, pe)) setup(sub);
            output.set_size(prev_output.num_samples(), prev_output.k(), sequence_dim, embedding_dim);
            tt::add(output, prev_output, pe);
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            auto& prev_grad = sub.get_gradient_input();
            tt::add(prev_grad, prev_grad, gradient_input);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        const tensor& get_positional_encodings() const { return pe; }
        tensor& get_positional_encodings() { return pe; }

        friend void serialize(const positional_encodings_& /* item */, std::ostream& out)
        {
            serialize("positional_encodings_", out);
        }
        friend void deserialize(positional_encodings_& /* item */, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "positional_encodings_")
                throw serialization_error("Unexpected version '" + version + "' found while deserializing dlib::positional_encodings_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const positional_encodings_& /*item*/)
        {
            out << "positional_encodings";
            return out;
        }
        friend void to_xml(const positional_encodings_& /*item*/, std::ostream& out) {
            out << "<positional_encodings />\n";
        }

    private:
        resizable_tensor params; // unused
        resizable_tensor pe;
        unsigned long sequence_dim, embedding_dim;
    };

    template <typename SUBNET>
    using positional_encodings = add_layer<positional_encodings_, SUBNET>;

    enum linear_bias_mode { LINEAR_HAS_BIAS = 0, LINEAR_NO_BIAS = 1 };

    template <
        unsigned long num_outputs_,
        linear_bias_mode bias_mode_
        >
    class linear_
    {
        static_assert(num_outputs_ > 0, "The number of outputs from a linear_ layer must be > 0");

    public:
        linear_() :
            num_outputs(num_outputs_),
            num_inputs(0),
            learning_rate_multiplier(1),
            bias_mode(bias_mode_) {}

        double get_learning_rate_multiplier() const { return learning_rate_multiplier; }
        void set_learning_rate_multiplier(double val) { learning_rate_multiplier = val; }

        unsigned long get_num_inputs() const { return num_inputs; }
        unsigned long get_num_outputs() const { return num_outputs; }
        void set_num_outputs(long num)
        {
            DLIB_CASSERT(num > 0);
            if (num != (long)num_outputs)
            {
                DLIB_CASSERT(get_layer_params().size() == 0,
                    "You can't change the number of filters in linear_ if the parameter tensor has already been allocated.");
                num_outputs = num;
            }
        }
        linear_bias_mode get_bias_mode() const { return bias_mode; }

        template <typename SUBNET>
        void setup(const SUBNET& sub)
        {
            num_inputs = sub.get_output().nc();
            if (bias_mode == LINEAR_HAS_BIAS)
                params.set_size(num_inputs + 1, num_outputs);
            else
                params.set_size(num_inputs, num_outputs);

            dlib::rand rnd(std::rand());
            randomize_parameters(params, num_inputs + num_outputs, rnd);            
            weights = alias_tensor(num_inputs, num_outputs);

            if (bias_mode == LINEAR_HAS_BIAS) {
                biases = alias_tensor(1, num_outputs);
                biases(params, weights.size()) = 0;
            }
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            const auto& prev_output = sub.get_output();
            DLIB_CASSERT((long)num_inputs == sub.get_output().nc(),
                "The size of the input tensor to this linear layer doesn't match the size the linear layer was trained with.");
            output.set_size(prev_output.num_samples(), prev_output.k(), prev_output.nr(), num_outputs);
            
            auto o = alias_tensor(output.num_samples() * output.k() * output.nr(), num_outputs)(output, 0);
            auto so = alias_tensor(prev_output.num_samples() * prev_output.k() * prev_output.nr(), num_inputs)(prev_output, 0);

            auto w = weights(params, 0);
            tt::gemm(0, (tensor&)o, 1, so, false, w, false, tt::operation_mode::CHANNEL_WISE);
            
            if (bias_mode == LINEAR_HAS_BIAS)
            {
                auto b = biases(params, weights.size());
                tt::add(1, (tensor&)o, 1, b);
            }
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& params_grad)
        {
            auto gi = alias_tensor(gradient_input.num_samples() * gradient_input.k() * gradient_input.nr(), num_outputs)(gradient_input, 0);
            if (learning_rate_multiplier != 0)
            {
                const auto& prev_output = sub.get_output();
                auto pw = weights(params_grad, 0);
                auto so = alias_tensor(prev_output.num_samples() * prev_output.k() * prev_output.nr(), num_inputs)(prev_output, 0);
                tt::gemm(0, pw, learning_rate_multiplier, so, true, gi, false, tt::operation_mode::CHANNEL_WISE);

                if (bias_mode == LINEAR_HAS_BIAS)
                {
                    auto pb = biases(params_grad, weights.size());
                    tt::assign_bias_gradient(pb, gi);
                }
            }
            
            const auto& prev_gradient = sub.get_gradient_input();
            auto sgi = alias_tensor(prev_gradient.num_samples() * prev_gradient.k() * prev_gradient.nr(), num_inputs)(prev_gradient, 0);
            auto w = weights(params, 0);
            tt::gemm(1, (tensor&)sgi, 1, gi, false, w, true, tt::operation_mode::CHANNEL_WISE);
        }

        alias_tensor_instance get_weights() { return weights(params, 0); }
        alias_tensor_const_instance get_weights() const { return weights(params, 0); }
        alias_tensor_instance get_biases()
        {
            static_assert(bias_mode == LINEAR_HAS_BIAS, "This linear_ layer doesn't have a bias vector "
                "to be retrieved, as per template parameter 'bias_mode'.");
            return biases(params, weights.size());
        }
        alias_tensor_const_instance get_biases() const
        {
            static_assert(bias_mode == LINEAR_HAS_BIAS, "This linear_ layer doesn't have a bias vector "
                "to be retrieved, as per template parameter 'bias_mode'.");
            return biases(params, weights.size());
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const linear_& item, std::ostream& out)
        {
            serialize("linear_", out);
            serialize(item.num_outputs, out);
            serialize(item.num_inputs, out);
            serialize(item.params, out);
            serialize(item.weights, out);
            serialize(item.biases, out);
            serialize(item.bias_mode, out);
            serialize(item.learning_rate_multiplier, out);
        }

        friend void deserialize(linear_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version == "linear_")
            {
                deserialize(item.num_outputs, in);
                deserialize(item.num_inputs, in);
                deserialize(item.params, in);
                deserialize(item.weights, in);
                deserialize(item.biases, in);
                deserialize(item.bias_mode, in);
                if (bias_mode_ != item.bias_mode) throw serialization_error("Wrong 'bias_mode' found while deserializing dlib::linear_");
                deserialize(item.learning_rate_multiplier, in);
            }
            else
            {
                throw serialization_error("Unexpected version '" + version + "' found while deserializing dlib::linear_.");
            }
        }

        friend std::ostream& operator<<(std::ostream& out, const linear_& item)
        {
            out << "linear (num_outputs=" << item.num_outputs
                << " bias=" << ((item.bias_mode == LINEAR_HAS_BIAS) ? "yes" : "no") << ")";
            out << " learning_rate_mult=" << item.learning_rate_multiplier;
            return out;
        }

        friend void to_xml(const linear_& item, std::ostream& out)
        {
            out << "<linear"
                << " num_outputs='" << item.num_outputs << "'"
                << " bias='" << ((item.bias_mode == LINEAR_HAS_BIAS) ? "yes" : "no") << "'"
                << " learning_rate_mult='" << item.learning_rate_multiplier << "'>\n";
            out << mat(item.params);
            out << "</linear>\n";
        }

    private:
        unsigned long num_inputs;
        unsigned long num_outputs;
        double learning_rate_multiplier;
        unsigned long bias_mode;
        resizable_tensor params;
        alias_tensor weights, biases;
    };

    template <
        unsigned long num_outputs,
        typename SUBNET
        >
    using linear = add_layer<linear_<num_outputs, LINEAR_HAS_BIAS>, SUBNET>;

    template <
        unsigned long num_outputs,
        typename SUBNET
        >
    using linear_no_bias = add_layer<linear_<num_outputs, LINEAR_NO_BIAS>, SUBNET>;

    // ----------------------------------------------------------------------------------------
    template <unsigned long nb_heads_>
    class hsplit_
    {
    public:
        hsplit_() : num_heads(nb_heads_) {}

        template <typename SUBNET>
        void setup(const SUBNET& sub)
        {
            const auto& input = sub.get_output();
            DLIB_CASSERT(num_heads > 1 && input.nc() % num_heads == 0, "Input dimension must be divisible by number of heads");
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            const auto& prev = sub.get_output();
            output.set_size(prev.num_samples(), prev.k() * num_heads, prev.nr(), prev.nc() / num_heads);

            tt::reorg2(false, output, 1, num_heads, prev);
            //tt::split_columns(false, output, prev, num_heads);
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            auto& grad = sub.get_gradient_input();

            tt::reorg_gradient2(true, grad, 1, num_heads, gradient_input);
            //tt::merge_columns(true, grad, gradient_input);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const hsplit_& item, std::ostream& out)
        {
            serialize("hsplit_", out);
            serialize(item.num_heads, out);
        }
        friend void deserialize(hsplit_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "hsplit_")
                throw serialization_error("Unexpected version '" + version + "' found while deserializing dlib::hsplit_.");
            deserialize(item.num_heads, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const hsplit_& item)
        {
            out << "hsplit (" << "num_heads=" << item.num_heads << ")";
            return out;
        }
        friend void to_xml(const hsplit_& item, std::ostream& out)
        {
            out << "<hsplit num_heads='" << item.num_heads << "''>\n";
            out << "</hsplit>\n";
        }

    private:
        resizable_tensor params; // unused
        unsigned long num_heads;
    };

    template <unsigned long num_heads, typename SUBNET>
    using hsplit = add_layer<hsplit_<num_heads>, SUBNET>;

    class hstack_
    {
    public:
        hstack_() {}
        template <typename SUBNET>
        void setup(const SUBNET& /* sub */) {}

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            const auto& prev = sub.get_output();
            output.set_size(prev.num_samples(), 1, prev.nr(), prev.nc() * prev.k());

            tt::reorg_gradient2(false, output, 1, prev.k(), prev);
            //tt::merge_columns(false, output, prev);
        }
        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            auto& grad = sub.get_gradient_input();

            tt::reorg2(true, grad, 1, grad.k(), gradient_input);
            //tt::split_columns(true, grad, gradient_input, grad.k());
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const hstack_& /* item */, std::ostream& out)
        {
            serialize("hstack_", out);
        }
        friend void deserialize(hstack_& /* item */, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "hstack_")
                throw serialization_error("Unexpected version '" + version + "' found while deserializing dlib::hstack_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const hstack_& /* item */)
        {
            out << "hstack";
            return out;
        }
        friend void to_xml(const hstack_& /* item */, std::ostream& out)
        {
            out << "<hstack />\n";
        }
    private:
        resizable_tensor params; // unused
    };

    template <typename SUBNET>
    using hstack = add_layer<hstack_, SUBNET>;

    class transpose_ {
    public:
        transpose_() {}
        template <typename SUBNET> void setup(const SUBNET& /* sub */) {}

        template <typename SUBNET> void forward(const SUBNET& sub, resizable_tensor& output) {
            auto& prev = sub.get_output();

            output.set_size(prev.num_samples(), prev.k(), prev.nc(), prev.nr());
            tt::transpose(false, output, prev);
        }

        template <typename SUBNET> void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/) {
            auto& prev = sub.get_gradient_input();
            tt::transpose(true, prev, gradient_input);
        }

        inline dpoint map_input_to_output(dpoint p) const
        {
            dpoint temp_p;
            temp_p.x() = p.y();
            temp_p.y() = p.x();
            return temp_p;
        }
        inline dpoint map_output_to_input(dpoint p) const
        {
            dpoint temp_p;
            temp_p.x() = p.y();
            temp_p.y() = p.x();
            return temp_p;
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const transpose_& /* item */, std::ostream& out) {
            serialize("transpose_", out);
        }
        friend void deserialize(transpose_& /* item */, std::istream& in) {
            std::string version;
            deserialize(version, in);
            if (version != "transpose_")
                throw serialization_error("Unexpected version '" + version + "' found while deserializing dlib::transpose_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const transpose_& /* item */) {
            out << "transpose";
            return out;
        }
        friend void to_xml(const transpose_& /* item */, std::ostream& out) {
            out << "<transpose />\n";
        }

    private:
        dlib::resizable_tensor params; // unused
    };

    template <typename SUBNET> using transpose = add_layer<transpose_, SUBNET>;

    struct neg_infinity_tag {};
    struct zero_tag {};

    template<typename T>
    struct is_special_value : std::false_type {};
    template<>
    struct is_special_value<neg_infinity_tag> : std::true_type {};
    template<>
    struct is_special_value<zero_tag> : std::true_type {};

    template<long diag_, typename tag_, long num_ = 0, long den_ = 1>
    class tril_
    {
    public:
        tril_() : diag(diag_), diag_value(compute_diag_value()) {}

        template <typename SUBNET>
        void setup(const SUBNET& /*sub*/)
        {
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            auto& prev = sub.get_output();
            output.set_size(prev.num_samples(), prev.k(), prev.nr(), prev.nc());

            check_mask(prev);
            tt::multiply(false, output, prev, binary_mask);
            if (diag_value != 0.0f) tt::add(1, output, 1, output_mask);
        }
        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            auto& prev_grad = sub.get_gradient_input();
            tt::multiply(true, prev_grad, gradient_input, binary_mask);
        }

        inline dpoint map_input_to_output(const dpoint& p) const { return p; }
        inline dpoint map_output_to_input(const dpoint& p) const { return p; }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const tril_& item, std::ostream& out)
        {
            serialize("tril_", out);
            serialize(item.diag, out);
            serialize(item.diag_value, out);
        }
        friend void deserialize(tril_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "tril_")
                throw serialization_error("Unexpected version '" + version + "' found while deserializing dlib::tril_.");
            deserialize(item.diag, in);
            deserialize(item.diag_value, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const tril_& item)
        {
            out << "tril (diag=" << item.diag << ", diag_value=" << item.diag_value << ")";
            return out;
        }
        friend void to_xml(const tril_& item, std::ostream& out)
        {
            out << "<tril diag='" << item.diag << "' diag_value='" << item.diag_value << "'/>\n";
        }

    private:
        float compute_diag_value() const {
            if (std::is_same<tag_, neg_infinity_tag>::value)
                return -std::numeric_limits<float>::infinity();
            else if (std::is_same<tag_, zero_tag>::value)
                return 0.0f;
            else
                return static_cast<float>(num_) / static_cast<float>(den_);
        }

        void check_mask(const tensor& t)
        {
            if (!have_same_dimensions(binary_mask, t)) {
                binary_mask.copy_size(t);
                binary_mask = 1;
                if (diag_value != 0.0f) {
                    output_mask.copy_size(t);
                    output_mask = 0;
                }
                for (long s = 0; s < output_mask.num_samples(); ++s)
                {
                    for (long k = 0; k < output_mask.k(); ++k)
                    {
                        for (long r = 0; r < output_mask.nr(); ++r)
                        {
                            for (long c = std::max(r + diag + 1, 0L); c < output_mask.nc(); ++c)
                            {
                                if (diag_value != 0.0f) output_mask.host()[tensor_index(output_mask, s, k, r, c)] = diag_value;
                                binary_mask.host()[tensor_index(binary_mask, s, k, r, c)] = 0;
                            }
                        }
                    }
                }
            }
        }

        template <typename T>
        struct always_false : std::false_type {};

        resizable_tensor params; // unused
        resizable_tensor binary_mask, output_mask;
        long diag;
        float diag_value;
    };

    template <typename SUBNET>
    using tril = add_layer<tril_<0, zero_tag>, SUBNET>;

    template <typename SUBNET>
    using tril_mask = add_layer<tril_<0, neg_infinity_tag>, SUBNET>;

    template <long diag, long num, long den, typename SUBNET>
    using tril_diag = add_layer<tril_<diag, void, num, den>, SUBNET>;

    template <
        template<typename> class tag
    >
    class multm_prev_
    {
    public:
        const static unsigned long id = tag_id<tag>::id;

        multm_prev_() {}
        template <typename SUBNET> void setup(const SUBNET& /*sub*/) {}

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            auto& t1 = sub.get_output();
            auto& t2 = layer<tag>(sub).get_output();
            output.set_size(t1.num_samples(), t1.k(), t1.nr(), t2.nc());

            tt::gemm(0, output, 1, t1, false, t2, false, tt::operation_mode::PLANE_WISE);
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            auto& t1 = sub.get_output();
            auto& t2 = layer<tag>(sub).get_output();
            auto& prev = sub.get_gradient_input();
            auto& prev_tag = layer<tag>(sub).get_gradient_input();

            tt::gemm(1, prev, 1, gradient_input, false, t2, true, tt::operation_mode::PLANE_WISE);
            tt::gemm(1, prev_tag, 1, t1, true, gradient_input, false, tt::operation_mode::PLANE_WISE);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        inline dpoint map_input_to_output(const dpoint& p) const { return p; }
        inline dpoint map_output_to_input(const dpoint& p) const { return p; }

        friend void serialize(const multm_prev_& /*item*/, std::ostream& out)
        {
            serialize("multm_prev_", out);
        }
        friend void deserialize(multm_prev_& /*item*/, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "multm_prev_")
                throw serialization_error("Unexpected version '" + version + "' found while deserializing dlib::multm_prev_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const multm_prev_& /*item*/)
        {
            out << "multm_prev" << id;
            return out;
        }
        friend void to_xml(const multm_prev_& /*item*/, std::ostream& out)
        {
            out << "<multm_prev tag='" << id << "'/>\n";
        }

    private:
        resizable_tensor params; // unused
    };

    template <
        template<typename> class tag,
        typename SUBNET
    >
    using multm_prev = add_layer<multm_prev_<tag>, SUBNET>;

    template <typename SUBNET> using multm_prev1 = multm_prev<tag1, SUBNET>;
    template <typename SUBNET> using multm_prev2 = multm_prev<tag2, SUBNET>;
    template <typename SUBNET> using multm_prev3 = multm_prev<tag3, SUBNET>;
    template <typename SUBNET> using multm_prev4 = multm_prev<tag4, SUBNET>;
    template <typename SUBNET> using multm_prev5 = multm_prev<tag5, SUBNET>;
    template <typename SUBNET> using multm_prev6 = multm_prev<tag6, SUBNET>;
    template <typename SUBNET> using multm_prev7 = multm_prev<tag7, SUBNET>;
    template <typename SUBNET> using multm_prev8 = multm_prev<tag8, SUBNET>;
    template <typename SUBNET> using multm_prev9 = multm_prev<tag9, SUBNET>;
    template <typename SUBNET> using multm_prev10 = multm_prev<tag10, SUBNET>;
    using multm_prev1_ = multm_prev_<tag1>;
    using multm_prev2_ = multm_prev_<tag2>;
    using multm_prev3_ = multm_prev_<tag3>;
    using multm_prev4_ = multm_prev_<tag4>;
    using multm_prev5_ = multm_prev_<tag5>;
    using multm_prev6_ = multm_prev_<tag6>;
    using multm_prev7_ = multm_prev_<tag7>;
    using multm_prev8_ = multm_prev_<tag8>;
    using multm_prev9_ = multm_prev_<tag9>;
    using multm_prev10_ = multm_prev_<tag10>;

    template <unsigned long s_mode_>
    class softmax2_
    {
    public:
        softmax2_() {}

        template <typename SUBNET>
        void setup(const SUBNET& /*sub*/) {}

        void forward_inplace(const tensor& input, tensor& output)
        {
            tt::softmax(output, input, (tt::operation_mode)s_mode_);
        }

        void backward_inplace(
            const tensor& computed_output,
            const tensor& gradient_input,
            tensor& data_grad,
            tensor& /*params_grad*/
        )
        {
            tt::softmax_gradient(data_grad, computed_output, gradient_input, (tt::operation_mode)s_mode_);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const softmax2_& item, std::ostream& out)
        {
            serialize("softmax2_", out);
        }

        friend void deserialize(softmax2_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "softmax2_")
                throw serialization_error("Unexpected version '" + version + "' found while deserializing dlib::softmax2_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const softmax2_& item)
        {
            out << "softmax2 (mode=" << (s_mode_ == static_cast<unsigned long>(tt::operation_mode::CHANNEL_WISE) ? "channel_wise" : "plane_wise") << ")";
            return out;
        }

        friend void to_xml(const softmax2_& item, std::ostream& out)
        {
            out << "<softmax2 mode='" << (s_mode_ == static_cast<unsigned long>(tt::operation_mode::CHANNEL_WISE) ? "channel_wise" : "plane_wise") << "'/>\n";
        }

    private:
        resizable_tensor params; // unused
    };

    template <typename SUBNET>
    using softmax2 = add_layer<softmax2_<static_cast<unsigned long>(tt::operation_mode::CHANNEL_WISE)>, SUBNET>;

    template <typename SUBNET>
    using softmaxm = add_layer<softmax2_< static_cast<unsigned long>(tt::operation_mode::PLANE_WISE)>, SUBNET>;

// ----------------------------------------------------------------------------------------
    /* TO BE ADDED TO <loss.h> */ 

    class loss_cross_entropy_
    {
    public:

        typedef unsigned long training_label_type;
        typedef unsigned long output_label_type;        

        template <
            typename SUB_TYPE,
            typename label_iterator
        >
        void to_label(
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const
        {
            const tensor& output_tensor = sub.get_output();
            DLIB_CASSERT(sub.sample_expansion_factor() == 1);
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());

            const size_t num_samples = output_tensor.num_samples();
            const size_t num_channels = output_tensor.k();
            const size_t plane_size = output_tensor.nr() * output_tensor.nc();
            matrix<float> sums(output_tensor.nr(), output_tensor.nc());
            for (size_t s = 0; s < num_samples; ++s)
            {
                sums = 0.0f;
                for (size_t k = 0; k < num_channels; ++k)
                {
                    auto o_plane = alias_tensor(output_tensor.nr(), output_tensor.nc())(output_tensor, (s * num_channels + k) * plane_size);
                    sums += mat(o_plane);
                }
                *iter++ = index_of_max(sum_rows(sums));
            }
        }

        template <
            typename const_label_iterator,
            typename SUBNET
        >
        double compute_loss_value_and_gradient(
            const tensor& input_tensor,
            const_label_iterator truth,
            SUBNET& sub
        ) const
        {
            const float label_smoothing = 0.1f;
            const tensor& output_tensor = sub.get_output();
            tensor& grad = sub.get_gradient_input();

            DLIB_CASSERT(sub.sample_expansion_factor() == 1);
            DLIB_CASSERT(input_tensor.num_samples() != 0);
            DLIB_CASSERT(input_tensor.num_samples() % sub.sample_expansion_factor() == 0);
            DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples());
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
            DLIB_CASSERT(have_same_dimensions(output_tensor, grad));

            tt::softmax(grad, output_tensor, tt::operation_mode::PLANE_WISE);

            double loss = 0.0;
            const size_t ns = grad.num_samples(), nk = grad.k(), nr = grad.nr(), nc = grad.nc();
            const float scale = 1.0f / ns;
            const size_t plane_size = nr * nc;

            for (size_t s = 0; s < ns; ++s)
            {
                const long y = (long)*truth++;
                DLIB_CASSERT(y < nc);
                                
                for (size_t k = 0; k < nk; ++k)
                {                    
                    auto grad_plane = alias_tensor(nr, nc)(grad, (s * nk + k) * plane_size);
                    float prob_y = 0.0f;

                    for (size_t r = 0; r < nr; ++r)
                    {
                        for (size_t c = 0; c < nc; ++c)
                        {
                            const long idx = r * nc + c;
                            const float prob = grad_plane.host()[idx];
                            float target = (c == y) ? (1.0f - label_smoothing) : (nc > 1 ? label_smoothing / (nc - 1) : label_smoothing);
                            prob_y += target * prob;
                            grad_plane.host()[idx] = scale * (prob - target);
                        }
                    }
                    loss += scale * -safe_log(prob_y / nr);
                }                
            }

            return loss;
        }

        friend void serialize(const loss_cross_entropy_&, std::ostream& out)
        {
            serialize("loss_cross_entropy_", out);
        }

        friend void deserialize(loss_cross_entropy_&, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "loss_cross_entropy_")
                throw serialization_error("Unexpected version found while deserializing dlib::loss_cross_entropy_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_cross_entropy_&)
        {
            out << "loss_cross_entropy";
            return out;
        }

        friend void to_xml(const loss_cross_entropy_& /*item*/, std::ostream& out)
        {
            out << "<loss_cross_entropy />";
        }

    };

    template <typename SUBNET>
    using loss_cross_entropy = add_loss_layer<loss_cross_entropy_, SUBNET>;

    /* TO BE ADDED TO <input_abstract.h> */
// ----------------------------------------------------------------------------------------

    //template <typename T, size_t vocab_size_>
    //class one_hot_
    //{
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is a custom input layer that converts an integer stream into a one-hot tensor.

            THREAD SAFETY
                to_tensor() must be thread safe.  That is, multiple threads must be able to
                make calls to to_tensor() on a single instance of this object at the same
                time.
        !*/
    //public:
    //    typedef matrix<int, 0, 1> input_type;

    //    one_hot_(
    //    );
        /*!
            ensures
                - Default constructs this object. This function is not required to do
                  anything in particular but it must exist, that is, it is required that
                  layer objects be default constructable.
        !*/

    //    template <typename forward_iterator>
    //    void to_tensor(
    //        forward_iterator ibegin,
    //        forward_iterator iend,
    //        resizable_tensor& data
    //    ) const;
        /*!
            requires
                - [ibegin, iend) is an iterator range over input_type objects.
                - std::distance(ibegin,iend) > 0
                - The input range should contain matrices that all have the same
                  dimensions.
            ensures
                - Converts the iterator range into a one-hot tensor and stores it into #data.
                - #data.num_samples() == std::distance(ibegin,iend)
                - #data.k() == 1
                - #data.nr() == nr
                - #data.nc() == vocab_size_
                  where nr is the number of rows in the input matrices.
        !*/

    //    friend void serialize(const one_hot_& item, std::ostream& out);
    //    friend void deserialize(one_hot_& item, std::istream& in);
        /*!
            provides serialization support
        !*/

    //    friend std::ostream& operator<<(std::ostream& out, const one_hot_& item);
        /*!
            print a string describing this layer.
        !*/

    //   friend void to_xml(const one_hot_& item, std::ostream& out);
        /*!
            This function is optional, but required if you want to print your networks with
            net_to_xml().  Therefore, to_xml() prints a layer as XML.
        !*/
    //};*/

    // ----------------------------------------------------------------------------------------

    template <typename T, size_t VOCAB_SIZE>
    class one_hot
    {
    public:
        typedef matrix<int, 0, 1> input_type;

        one_hot() {}        

        template <typename forward_iterator>
        void to_tensor(
            forward_iterator ibegin,
            forward_iterator iend,
            resizable_tensor& data
        ) const
        {
            DLIB_CASSERT(std::distance(ibegin, iend) > 0);
            const auto nr = ibegin->nr();
            const auto nc = ibegin->nc();
            DLIB_CASSERT(nr > 0);
            DLIB_CASSERT(nc == 1);

            data.set_size(std::distance(ibegin, iend), 1, nr, VOCAB_SIZE);
            data = 0.0f;

            for (auto s = ibegin; s != iend; ++s)
            {
                for (long r = 0; r < nr; ++r)
                {
                    const long c = (*s)(r, 0);
                    data.host()[tensor_index(data, s - ibegin, 0, r, c)] = 1.0f;
                }
            }
        }

        friend void serialize(const one_hot& item, std::ostream& out)
        {
            serialize("one_hot", out);
        }

        friend void deserialize(one_hot& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "one_hot_")
                throw serialization_error("Unexpected version found while deserializing dlib::one_hot.");
        }

        friend std::ostream& operator<<(std::ostream& out, const one_hot& item)
        {
            out << "one_hot";
            return out;
        }

        friend void to_xml(const one_hot& item, std::ostream& out)
        {
            out << "<one_hot/>";
        }
    };

    template <typename T, size_t VOCAB_SIZE, size_t EMB_SIZE>
    class static_embeddings
    {
    public:
        typedef matrix<int, 0, 1> input_type;

        static_embeddings()
        {
            embeddings_.set_size(VOCAB_SIZE, EMB_SIZE);
            tt::tensor_rand rnd(std::rand());
            rnd.fill_gaussian(embeddings_);
        }

        template <typename forward_iterator>
        void to_tensor(
            forward_iterator ibegin,
            forward_iterator iend,
            resizable_tensor& data
        ) const
        {
            DLIB_CASSERT(std::distance(ibegin, iend) > 0);
            const auto nr = ibegin->nr();
            const auto nc = ibegin->nc();
            DLIB_CASSERT(nr > 0);
            DLIB_CASSERT(nc == 1);

            data.set_size(std::distance(ibegin, iend), 1, nr, EMB_SIZE);

            for (auto s = ibegin; s != iend; ++s)
            {
                for (long r = 0; r < nr; ++r)
                {
                    const long idx = (*s)(r, 0);
                    for (long c = 0; c < EMB_SIZE; ++c)
                    {
                        data.host()[tensor_index(data, s - ibegin, 0, r, c)] = embeddings_.host()[tensor_index(embeddings_, idx, c, 0, 0)];
                    }
                }
            }
        }

        friend void serialize(const static_embeddings& item, std::ostream& out)
        {
            serialize("static_embeddings", out);
            serialize(item.embeddings_, out);
        }

        friend void deserialize(static_embeddings& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "static_embeddings")
                throw serialization_error("Unexpected version found while deserializing dlib::static_embeddings.");
            deserialize(item.embeddings_, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const static_embeddings& item)
        {
            out << "static_embeddings";
            return out;
        }

        friend void to_xml(const static_embeddings& item, std::ostream& out)
        {
            out << "<static_embeddings/>";
        }

    private:
        resizable_tensor embeddings_;
    };
}

#endif // DlibExt
