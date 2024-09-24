/*
    This is an example illustrating the implementation of a Very Small Language Model (VSLM)
    using the deep learning tools from the dlib C++ Library. The program, named ERNIE
    (Efficient Rapid Neural Intelligence Engine), demonstrates how to extend dlib's
    capabilities to handle natural language processing tasks, specifically focusing on
    transformer-based architectures.

    Key features of this implementation include:
    - Custom layers designed for matrix-based processing of inputs, optimized for dlib's
      tensor structure.
    - Specialized input layers for LLM, including embedding injection and positional encoding.
    - A complete example of training and using a language model.
    - Benchmarking and testing suite, including a "Shakespeare test" that showcases the
      model's ability to generate text in the style of the famous playwright.

    The program is structured into several main components:
    1. Vocabulary training and testing
    2. Model training
    3. Text generation / chat mode
    4. Benchmarking and unit tests

    A notable feature is the Shakespeare test, where the model is trained on a sample of
    Shakespeare's text and then generates new text in a similar style. This demonstrates
    the model's capacity to learn and reproduce complex language patterns.

    This example serves as a starting point for implementing transformer-based language
    models using dlib, showcasing how to adapt the library for advanced NLP tasks. It
    provides insights into creating custom layers, handling sequential data, and
    implementing attention mechanisms within the dlib framework.
*/
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <random>
#include <csignal>
#include <deque>
#include <regex>
#include <algorithm>
#include <io.h>
#include <fcntl.h>
#include <atomic>
#include <boost/program_options.hpp>
#include <dlib/dnn.h>
#include <dlib/matrix.h>
#include <dlib/matrix/matrix_utilities.h>
#include <dlib/data_io.h>
#include <sentencepiece_trainer.h>
#include <sentencepiece_processor.h>
#ifdef DLIB_USE_CUDA
#include "cuda_dlib_ext.cuh"
#include <cuda_runtime.h>
#include "cublas_v2.h"
#endif // DLIB_USE_CUDA
#include "advanced_tokenizer.hpp"
#include "data_fr.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

using namespace std;
using namespace dlib;

// Global parameters for the Transformer network
const int vocab_size = 20000;                                           // Size of the vocabulary
const int sequence_size = 24;                                           // Length of the sequence
const int number_of_heads = 8;                                          // Number of attention heads
const int number_of_blocks = 4;                                         // Number of transformer blocks
const int embedding_size = (64 / number_of_heads) * number_of_heads;   // Size of the embedding
const int bos_id = 0, eos_id = 1, unk_id = 2, pad_id = 3;

// Other global parameters
string vocabulary_prefix = "ernie.en-fr.ung.20k", language_model = "ernie_vslm_v1.dat";
std::unique_ptr<advanced_tokenizer> tokenizer_;

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

void configure_console() {
    SetConsoleOutputCP(CP_UTF8);
    int res = _setmode(_fileno(stdout), _O_TEXT);
    if (res == -1) cerr << "Cannot set mode" << endl;
    cout.imbue(std::locale("en_US.UTF-8"));    
}

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

namespace utils {
    std::string replace_html_entities(const std::string& input) {
        static const std::unordered_map<std::string, std::string> htmlEntities = {
            {"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"}, {"&quot;", "\""}, {"&apos;", "'"}, {"&nbsp;", " "}, {"&iexcl;", "¡"}, {"&cent;", "¢"},
            {"&pound;", "£"}, {"&curren;", "¤"}, {"&yen;", "¥"}, {"&brvbar;", "¦"}, {"&sect;", "§"}, {"&uml;", "¨"}, {"&copy;", "©"},
            {"&ordf;", "ª"}, {"&laquo;", "«"}, {"&not;", "¬"}, {"&shy;", "\u00AD"}, {"&reg;", "®"}, {"&macr;", "¯"}, {"&deg;", "°"},
            {"&plusmn;", "±"}, {"&sup2;", "²"}, {"&sup3;", "³"}, {"&acute;", "´"}, {"&micro;", "µ"}, {"&para;", "¶"}, {"&middot;", "·"},
            {"&cedil;", "¸"}, {"&sup1;", "¹"}, {"&ordm;", "º"}, {"&raquo;", "»"}, {"&frac14;", "¼"}, {"&frac12;", "½"}, {"&frac34;", "¾"},
            {"&iquest;", "¿"}, {"&times;", "×"}, {"&divide;", "÷"}, {"&ETH;", "Ð"}, {"&eth;", "ð"}, {"&THORN;", "Þ"}, {"&thorn;", "þ"},
            {"&szlig;", "ß"}, {"&Agrave;", "À"}, {"&agrave;", "à"}, {"&Aacute;", "Á"}, {"&aacute;", "á"}, {"&Acirc;", "Â"}, {"&acirc;", "â"},
            {"&Atilde;", "Ã"}, {"&atilde;", "ã"}, {"&Auml;", "Ä"}, {"&auml;", "ä"}, {"&Aring;", "Å"}, {"&aring;", "å"}, {"&AElig;", "Æ"},
            {"&aelig;", "æ"}, {"&Ccedil;", "Ç"}, {"&ccedil;", "ç"}, {"&Egrave;", "È"}, {"&egrave;", "è"}, {"&Eacute;", "É"}, {"&eacute;", "é"},
            {"&Ecirc;", "Ê"}, {"&ecirc;", "ê"}, {"&Euml;", "Ë"}, {"&euml;", "ë"}, {"&Igrave;", "Ì"}, {"&igrave;", "ì"}, {"&Iacute;", "Í"},
            {"&iacute;", "í"}, {"&Icirc;", "Î"}, {"&icirc;", "î"}, {"&Iuml;", "Ï"}, {"&iuml;", "ï"}, {"&Ntilde;", "Ñ"}, {"&ntilde;", "ñ"},
            {"&Ograve;", "Ò"}, {"&ograve;", "ò"}, {"&Oacute;", "Ó"}, {"&oacute;", "ó"}, {"&Ocirc;", "Ô"}, {"&ocirc;", "ô"}, {"&Otilde;", "Õ"},
            {"&otilde;", "õ"}, {"&Ouml;", "Ö"}, {"&ouml;", "ö"}, {"&Oslash;", "Ø"}, {"&oslash;", "ø"}, {"&Ugrave;", "Ù"}, {"&ugrave;", "ù"},
            {"&Uacute;", "Ú"}, {"&uacute;", "ú"}, {"&Ucirc;", "Û"}, {"&ucirc;", "û"}, {"&Uuml;", "Ü"}, {"&uuml;", "ü"}, {"&Yacute;", "Ý"},
            {"&yacute;", "ý"}, {"&Yuml;", "Ÿ"}, {"&yuml;", "ÿ"}
        };

        std::string output;
        output.reserve(input.size());

        size_t lastPos = 0;
        size_t findPos = 0;

        while ((findPos = input.find('&', lastPos)) != std::string::npos) {
            output.append(input, lastPos, findPos - lastPos);
            auto endPos = input.find(';', findPos);
            if (endPos != std::string::npos) {
                std::string entity = input.substr(findPos, endPos - findPos + 1);
                auto it = htmlEntities.find(entity);
                if (it != htmlEntities.end()) {
                    output.append(it->second);
                }
                else {
                    output.append(entity);
                }
                lastPos = endPos + 1;
            }
            else {
                break;
            }
        }
        output.append(input, lastPos, std::string::npos);
        return output;
    }

    bool is_unicode(char32_t c) {
        return ((c & 0xFFFE) != 0xFFFE) && (c < 0x10FFFF);
    }

    bool is_surrogate(char32_t c) { return (c & 0xF800) == 0xD800; }
    bool is_high_surrogate(char32_t c) { return (c & 0xFC00) == 0xD800; }
    bool is_low_surrogate(char32_t c) { return (c & 0xFC00) == 0xDC00; }

    char32_t decode(const char*& first, const char* last, char32_t invalid = 0xFFFD) {
        if (first == last) return invalid;

        static const unsigned char nbytes[] = {
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 3, 3, 4, 0
        };
        unsigned char k, n = k = nbytes[(unsigned char)*first >> 3];
        if (!n) { ++first; return invalid; }

        static const unsigned char masks[] = { 0, 0x7F, 0x1F, 0x0F, 0x07 };
        char32_t c = (unsigned char)*first++ & masks[n];

        while (--n && (first != last) && ((signed char)*first < -0x40)) c = (c << 6) | ((unsigned char)*first++ & 0x3F);
        if (n != 0) return invalid;
        if (k != 1 + (c > 0x7F) + (c > 0x7FF) + (c > 0xFFFF)) return invalid;

        return is_unicode(c) && !is_surrogate(c) ? c : invalid;
    }

    bool is_utf8(const std::string& s) {
        return [](const char* first, const char* last) {
            if (first != last) {
                if ((last - first) > 2)
                    if (((unsigned char)first[0] == 0xEF) && ((unsigned char)first[1] == 0xBF) && ((unsigned char)first[2] == 0xBE))
                        first += 3;
                while (first != last)
                    if (decode(first, last, 0x10FFFF) == 0x10FFFF) return false;
            }
            return true;
        }
        (s.c_str(), s.c_str() + s.size());
    }

    void concatenate_files(const string& directory_path, const string& output_file = "raw_data.txt") {
        std::ofstream output(fs::current_path().string() + "/" + output_file, std::ios::binary);
        if (!output) {
            std::cerr << "Error opening output file: " << output_file << std::endl;
            return;
        }

        for (const auto& entry : fs::recursive_directory_iterator(directory_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                std::ifstream input(entry.path(), std::ios::binary);
                if (!input) {
                    std::cerr << "Error opening input file: " << entry.path() << std::endl;
                    continue;
                }
                cout << "parsing file: " << entry.path().string() << endl;
                output << input.rdbuf() << "\n";
            }
        }
    }

}
using utils::replace_html_entities;
using utils::is_utf8;
using utils::concatenate_files;

namespace dlib {
    namespace cpu {
        /* TO BE ADDED TO <cpu_dlib.cpp> */

        namespace ttimpl
        {
            void softmax2(
                const long num_locations,
                const long num_channels,
                tensor& dest,
                const tensor& src,
                size_t mode
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

                    if (mode == 0)
                    {
                        // Mode 0: softmax along the channel axis (k)
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
                            sum += std::numeric_limits<float>::epsilon();
                            for (long k = 0; k < num_channels; ++k)
                                dd[k * num_locations] /= sum;

                            ++ss;
                            ++dd;
                        }
                    }
                    else
                    {
                        // Mode 1: softmax along the spatial locations (nr x nc)
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
                                        d_channel[idx] = (1.0f / src.nc());
                                }
                                else
                                {
                                    float sum = 0.0f;
                                    for (long c = 0, idx = r * src.nc(); c < src.nc(); ++c, ++idx)
                                    {
                                        d_channel[idx] = std::exp(s_channel[idx] - max_val);
                                        sum += d_channel[idx];
                                    }
                                    sum += std::numeric_limits<float>::epsilon();
                                    for (long c = 0, idx = r * src.nc(); c < src.nc(); ++c, ++idx)
                                        d_channel[idx] /= sum;
                                }
                            }                           
                        }
                    }
                }
            }

            void softmax_gradient2(
                const long num_locations,
                const long num_channels,
                tensor& grad,
                const tensor& dest,
                const tensor& gradient_input,
                size_t mode
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
                    if (mode == 0) // softmax_mode::CHANNEL_WISE
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
                    else  // softmax_mode::PLANE_WISE
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

        void softmax2(
            tensor& dest,
            const tensor& src,
            size_t mode
        )
        {
            DLIB_CASSERT(have_same_dimensions(dest, src));
            ttimpl::softmax2(src.nr() * src.nc(), src.k(), dest, src, mode);
        }

        void softmax_gradient2(
            tensor& grad,
            const tensor& dest,
            const tensor& gradient_input,
            size_t mode
        )
        {
            DLIB_CASSERT(have_same_dimensions(grad, dest));
            DLIB_CASSERT(have_same_dimensions(grad, gradient_input));
            ttimpl::softmax_gradient2(grad.nr() * grad.nc(), grad.k(), grad, dest, gradient_input, mode);
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
            const tensor& emb
        )
        {
            DLIB_CASSERT(
                src.nr() > 0 &&
                emb.num_samples() > 0 &&
                emb.k() > 0 &&
                emb.nr() == 1 &&
                emb.nc() == 1,
                "\nsrc.num_samples(): " << src.num_samples() <<
                "\nsrc.k(): " << src.k() <<
                "\nsrc.nr(): " << src.nr() <<
                "\nsrc.nc(): " << src.nc() <<
                "\nemb.num_samples(): " << emb.num_samples() <<
                "\nemb.k(): " << emb.k() <<
                "\nemb.nr(): " << emb.nr() <<
                "\nemb.nc(): " << emb.nc()
            );

            long ns = dest.num_samples(), nk = dest.k(), nr = dest.nr(), nc = dest.nc();
            const float* src_data = src.host();
            float* dest_data = dest.host();
            const float* emb_data = emb.host();
            for (long s = 0; s < ns; ++s)
            {
                for (long k = 0; k < nk; ++k)
                {
                    for (long r = 0; r < nr; ++r)
                    {
                        const unsigned long token_idx = static_cast<unsigned long>(src_data[tensor_index(src, s, k, r, 0)]);
                        if (token_idx < emb.num_samples())
                        {
                            for (long c = 0; c < nc; ++c)
                                dest_data[tensor_index(dest, s, k, r, c)] = emb_data[tensor_index(emb, token_idx, c, 0, 0)];
                        }
                        else
                        {
                            for (long c = 0; c < nc; ++c)
                                dest_data[tensor_index(dest, s, k, r, c)] = 1;
                        }
                    }
                }
            }
        }
        
        void embeddings_gradient(
            const tensor& prev,
            const tensor& gradient_input,
            tensor& emb,
            double rate,
            bool scale
        )
        {
            DLIB_CASSERT(
                prev.nr() > 0 &&
                gradient_input.num_samples() == prev.num_samples() &&
                gradient_input.k() == prev.k() &&
                gradient_input.nr() == prev.nr() &&
                gradient_input.nc() == emb.k() &&
                emb.num_samples() > 0 &&
                emb.k() > 0 &&
                emb.nr() == 1 &&
                emb.nc() == 1,
                "\ngradient_input.num_samples(): " << gradient_input.num_samples() <<
                "\ngradient_input.k(): " << gradient_input.k() <<
                "\ngradient_input.nr(): " << gradient_input.nr() <<
                "\ngradient_input.nc(): " << gradient_input.nc() <<
                "\nprev.num_samples(): " << prev.num_samples() <<
                "\nprev.k(): " << prev.k() <<
                "\nprev.nr(): " << prev.nr() <<
                "\nprev.nc(): " << prev.nc() <<
                "\nemb.num_samples(): " << emb.num_samples() <<
                "\nemb.k(): " << emb.k() <<
                "\nemb.nr(): " << emb.nr() <<
                "\nemb.nc(): " << emb.nc()
            );

            const float* prev_data = prev.host();
            const float* gradient_input_data = gradient_input.host();
            float* emb_data = emb.host();
            long ns = gradient_input.num_samples(), nk = gradient_input.k();
            long nr = gradient_input.nr(), nc = gradient_input.nc();

            std::vector<std::mutex> embedding_mutexes(emb.num_samples());
            std::unordered_map<unsigned long, unsigned long> token_freq;
            if (scale)
            {
                for (long k = 0; k < nk; ++k)
                {
                    for (long s = 0; s < ns; ++s)
                    {
                        for (long r = 0; r < nr; ++r)
                        {
                            const unsigned long token_idx = static_cast<unsigned long>(prev_data[tensor_index(prev, s, k, r, 0)]);
                            if (token_idx < emb.num_samples())
                            {
                                auto it = token_freq.find(token_idx);
                                if (it != token_freq.end()) token_freq[token_idx] += 1;
                                else token_freq[token_idx] = 1;
                            }
                        }
                    }
                }
            }
            parallel_for(0, ns, [&](long s)
            {
                for (long k = 0; k < nk; ++k)
                {
                    for (long r = 0; r < nr; ++r)
                    {
                        const unsigned long token_idx = static_cast<unsigned long>(prev_data[tensor_index(prev, s, k, r, 0)]);
                        if (token_idx < emb.num_samples())
                        {
                            float freq_scale = 1.0f;
                            if (scale)
                            {
                                auto it = token_freq.find(token_idx);
                                if (it != token_freq.end()) freq_scale = std::min(0.1f, std::max(2.0f * (1.0f / it->second), 1.0f));
                            }
                            std::lock_guard<std::mutex> lock(embedding_mutexes[token_idx]);
                            for (long c = 0; c < nc; ++c)
                            {
                                const float gradient = gradient_input_data[tensor_index(gradient_input, s, k, r, c)];
                                emb_data[tensor_index(emb, token_idx, c, 0, 0)] -= (gradient * rate * freq_scale);
                            }
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

        void c_gemm(
            float beta,
            tensor& dest,
            float alpha,
            const tensor& lhs,
            bool trans_lhs,
            const tensor& rhs,
            bool trans_rhs
        )
        {
            const auto transa = trans_lhs ? CUBLAS_OP_T : CUBLAS_OP_N;
            const auto transb = trans_rhs ? CUBLAS_OP_T : CUBLAS_OP_N;

            long num_samples = std::max({ lhs.num_samples(), rhs.num_samples(), dest.num_samples() });
            long num_channels = std::max({ lhs.k(), rhs.k(), dest.k() });

            auto is_matrix = [](const auto& tensor) {
                return (tensor.num_samples() == 1 && tensor.k() == 1) ||
                    (tensor.nr() == 1 && tensor.nc() == 1);
            };
            const bool lhs_is_matrix = is_matrix(lhs), rhs_is_matrix = is_matrix(rhs), dest_is_matrix = is_matrix(dest);

            if (lhs_is_matrix && rhs_is_matrix && dest_is_matrix) {
                num_samples = num_channels = 1;
            }
            else {
                auto adjust = [&](const auto& tensor) {
                    if (!is_matrix(tensor)) {
                        if (tensor.num_samples() < num_samples) num_samples = tensor.num_samples();
                        if (tensor.k() < num_channels) num_channels = tensor.k();
                    }
                };
                adjust(lhs);
                adjust(rhs);
                adjust(dest);
            }

            long lhs_rows = (lhs_is_matrix && lhs.num_samples() > 1) ? lhs.num_samples() : lhs.nr();
            long lhs_cols = (lhs_is_matrix && lhs.k() > 1) ? lhs.k() : lhs.nc();
            long rhs_rows = (rhs_is_matrix && rhs.num_samples() > 1) ? rhs.num_samples() : rhs.nr();
            long rhs_cols = (rhs_is_matrix && rhs.k() > 1) ? rhs.k() : rhs.nc();
            long dest_rows = (dest_is_matrix && dest.num_samples() > 1) ? dest.num_samples() : dest.nr();
            long dest_cols = (dest_is_matrix && dest.k() > 1) ? dest.k() : dest.nc();

            const size_t lhs_plane_size = lhs_rows * lhs_cols;
            const size_t rhs_plane_size = rhs_rows * rhs_cols;
            const size_t dest_plane_size = dest_rows * dest_cols;

            for (long b = 0; b < num_samples; ++b)
            {
                for (long c = 0; c < num_channels; ++c)
                {
                    auto lhs_slice = lhs_is_matrix ? lhs.device() :
                        lhs.device() + (b * num_channels + c) * lhs_plane_size;
                    auto rhs_slice = rhs_is_matrix ? rhs.device() :
                        rhs.device() + (b * num_channels + c) * rhs_plane_size;
                    auto dest_slice = dest_is_matrix ? dest.device() :
                        dest.device() + (b * num_channels + c) * dest_plane_size;
                    const int k = trans_rhs ? rhs_cols : rhs_rows;

                    cublasSgemm(
                        context(), transb, transa, dest_cols, dest_rows, k,
                        &alpha, rhs_slice, rhs_cols, lhs_slice, lhs_cols,
                        &beta, dest_slice, dest_cols
                    );
                }
            }
        }
    }
#endif

    namespace tt {
/* TO BE ADDED TO <tensor_tools.h> */
// -----------------------------------------------------------------------------------
        void embeddings(
            resizable_tensor& dest,
            const tensor& src,
            const tensor& emb
        );
        /*!
            requires
                - src.nr() > 0
                - emb.num_samples() > 0
                - emb.k() > 0
                - emb.nr() == 1
                - emb.nc() == 1
            ensures
                - Projects tokens from the input tensor `src` into embeddings stored in `emb`.
                - The resulting embeddings are stored in the `dest` tensor.
                - For all valid s, k, r, c:
                    - If src(s,k,r,0) < emb.num_samples():
                        - #dest(s,k,r,c) == emb(src(s,k,r,0), c, 0, 0)
                    - Else:
                        - #dest(s,k,r,c) == 0
        */

        void embeddings_gradient(
            const tensor& prev,
            const tensor& gradient_input,
            tensor& emb,
            double rate,
            bool scale
        );
        /*!
            requires
                - prev.nr() > 0
                - gradient_input.num_samples() == prev.num_samples()
                - gradient_input.k() == prev.k()
                - gradient_input.nr() == prev.nr()
                - gradient_input.nc() == emb.k()
                - emb.num_samples() > 0
                - emb.k() > 0
                - emb.nr() == 1
                - emb.nc() == 1
            ensures
                - Updates the `emb` tensor based on the gradients.
                - If scale is true, scales the gradient by the frequency of the tokens.
        */
        
        void apply_positional_encoding(
            const tensor& pe,
            const tensor& input,
            tensor& output
        );
        /*!
            requires
                - pe.num_samples() == 1
                - pe.k() == 1
                - pe.nr() == input.nr()
                - pe.nc() == input.nc()
                - have_same_dimensions(output, input)
            ensures
                - Applies the positional encoding stored in pe to the input tensor and stores the result in output.
                - The positional encoding is applied to all channels of the input tensor.
                - For all valid s, k, r, c:
                    - #output(s,k,r,c) == pe(r,c)
        !*/

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

/* TO BE ADDED TO <tensor_tools.cpp> */
// ----------------------------------------------------------------------------------------

        void embeddings(
            resizable_tensor& dest,
            const tensor& src,
            const tensor& emb
        )
        {
#ifdef DLIB_USE_CUDA
            cuda::embeddings(dest, src, emb);
#else
            cpu::embeddings(dest, src, emb);
#endif
        }

        void embeddings_gradient(
            const tensor& prev,
            const tensor& gradient_input,
            tensor& emb,
            double rate,
            bool scale
        )
        {
#ifdef DLIB_USE_CUDA
            cpu::embeddings_gradient(prev, gradient_input, emb, rate, scale);
#else
            cpu::embeddings_gradient(prev, gradient_input, emb, rate, scale);
#endif
        }

        void softmax2(
            tensor& dest,
            const tensor& src,
            size_t s_mode = 0
        )
        {
#ifdef DLIB_USE_CUDA
            cpu::softmax2(dest, src, s_mode);
#else
            cpu::softmax2(dest, src, s_mode);
#endif
        }

        void softmax_gradient2(
            tensor& grad,
            const tensor& dest,
            const tensor& gradient_input,
            size_t s_mode = 0
        )
        {
#ifdef DLIB_USE_CUDA
            cpu::softmax_gradient2(grad, dest, gradient_input, s_mode);
#else
            cpu::softmax_gradient2(grad, dest, gradient_input, s_mode);
#endif
        }

        void c_gemm(
            float beta,
            tensor& dest,
            float alpha,
            const tensor& lhs,
            bool trans_lhs,
            const tensor& rhs,
            bool trans_rhs
        )
        {
#ifdef DLIB_USE_CUDA
            cuda::c_gemm(beta, dest, alpha, lhs, trans_lhs, rhs, trans_rhs);
#else
            auto is_matrix = [](const auto& tensor) {
                return (tensor.num_samples() == 1 && tensor.k() == 1) ||
                    (tensor.nr() == 1 && tensor.nc() == 1);
            };

            long num_samples = std::max({ lhs.num_samples(), rhs.num_samples(), dest.num_samples() });
            long num_channels = std::max({ lhs.k(), rhs.k(), dest.k() });
            const bool lhs_is_matrix = is_matrix(lhs), rhs_is_matrix = is_matrix(rhs), dest_is_matrix = is_matrix(dest);

            if (lhs_is_matrix && rhs_is_matrix && dest_is_matrix) {
                num_samples = num_channels = 1;
            }
            else
            {
                auto adjust = [&](const auto& tensor) {
                    if (!is_matrix(tensor)) {
                        if (tensor.num_samples() < num_samples) num_samples = tensor.num_samples();
                        if (tensor.k() < num_channels) num_channels = tensor.k();
                    }
                };
                adjust(lhs);
                adjust(rhs);
                adjust(dest);                
            }

            long lhs_rows = (lhs_is_matrix && lhs.num_samples() > 1) ? lhs.num_samples() : lhs.nr();
            long lhs_cols = (lhs_is_matrix && lhs.k() > 1) ? lhs.k() : lhs.nc();
            long rhs_rows = (rhs_is_matrix && rhs.num_samples() > 1) ? rhs.num_samples() : rhs.nr();
            long rhs_cols = (rhs_is_matrix && rhs.k() > 1) ? rhs.k() : rhs.nc();
            long dest_rows = (dest_is_matrix && dest.num_samples() > 1) ? dest.num_samples() : dest.nr();
            long dest_cols = (dest_is_matrix && dest.k() > 1) ? dest.k() : dest.nc();

            /*if (trans_lhs) std::swap(lhs_rows, lhs_cols);
            if (trans_rhs) std::swap(rhs_rows, rhs_cols);

            DLIB_CASSERT((!rhs_is_matrix && num_samples == rhs.num_samples()) || rhs_is_matrix, "Batch dimensions must match");
            DLIB_CASSERT((!rhs_is_matrix && num_channels == rhs.k()) || rhs_is_matrix, "Channel dimensions must match");
            DLIB_CASSERT(lhs_cols == rhs_rows, "Inner dimensions must match for matrix multiplication");
            DLIB_CASSERT((!dest_is_matrix && 
                num_samples == dest.num_samples() &&
                num_channels == dest.k() &&
                lhs_rows == dest.nr() &&
                rhs_cols == dest.nc()) || (dest_is_matrix &&
                ((dest.num_samples() == lhs_rows && dest.k() == rhs_cols) ||
                 (dest.nr() == lhs_rows && dest.nc() == rhs_cols))), "Incompatible destination tensor");

            if (trans_lhs) std::swap(lhs_rows, lhs_cols);
            if (trans_rhs) std::swap(rhs_rows, rhs_cols);*/

            const size_t lhs_plane_size = lhs_rows * lhs_cols;
            const size_t rhs_plane_size = rhs_rows * rhs_cols;
            const size_t dest_plane_size = dest_rows * dest_cols;

            for (long b = 0; b < num_samples; ++b)
            {
                for (long c = 0; c < num_channels; ++c)
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
    template <typename SUBNET>
    using dropout_10 = add_layer<dropout_rate_<10>, SUBNET>;

    // ----------------------------------------------------------------------------------------
    /* TO BE ADDED TO <layers.h> */

    template<unsigned long num_embeddings_, unsigned long embedding_dim_>
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

        template <typename SUBNET>
        void setup(const SUBNET& sub)
        {
            embs.set_size(num_embeddings, embedding_dim);
            dlib::rand rnd(std::rand());
            randomize_parameters(embs, num_embeddings + embedding_dim, rnd);
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
            if (learning_rate_multiplier != 0) {
                auto& prev_src = sub.get_output();
                tt::embeddings_gradient(prev_src, gradient_input, embs, learning_rate_multiplier, scale_by_freq);
            }
            // As the embeddings_ layer is positioned just after the input, 
            // there is no need to forward the gradient received
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
        friend void to_xml(const embeddings_& item, std::ostream& out) {
            out << "<embeddings num_embeddings='" << item.num_embeddings
                << "' embedding_dim='" << item.embedding_dim
                << "' learning_rate_mult='"
                << item.learning_rate_multiplier << "'>\n";
            out << mat(item.embs);
            out << "</embeddings>\n";
        }

    private:
        resizable_tensor params; // unused
        resizable_tensor embs;
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
            if (!have_same_dimensions(pe, prev_output)) setup(sub);
            
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

        friend void serialize(const positional_encodings_& /*item*/, std::ostream& out)
        {
            serialize("positional_encodings_", out);
        }
        friend void deserialize(positional_encodings_& /*item*/, std::istream& in)
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

    template <unsigned long nb_embeddings, unsigned long embedding_length, typename SUBNET>
    using positional_embeddings = positional_encodings<htan<embeddings<nb_embeddings, embedding_length, tag10<SUBNET>>>>;

    enum linear_bias_mode { LINEAR_HAS_BIAS = 0, LINEAR_NO_BIAS = 1 };
    struct num_linear_outputs
    {
        num_linear_outputs(unsigned long n) : num_outputs(n) {}
        unsigned long num_outputs;
    };
    template <unsigned long num_outputs_, linear_bias_mode bias_mode_>
    class linear_
    {
        static_assert(num_outputs_ > 0, "The number of outputs from a linear_ layer must be > 0");

    public:
        linear_(num_linear_outputs o) :
            num_outputs(o.num_outputs),
            num_inputs(0),
            learning_rate_multiplier(1),
            bias_mode(bias_mode_) {}
        linear_() : linear_(num_linear_outputs(num_outputs_)) {}

        double get_learning_rate_multiplier() const { return learning_rate_multiplier; }
        void set_learning_rate_multiplier(double val) { learning_rate_multiplier = val; }

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
            DLIB_CASSERT(num_inputs > 0, "The input to a linear layer must have a non-zero number of rows");
            DLIB_CASSERT(num_outputs > 0, "The number of outputs for a linear layer must be > 0");

            params.set_size(num_inputs + (bias_mode_ == LINEAR_HAS_BIAS ? 1 : 0), num_outputs);
            dlib::rand rnd;
            randomize_parameters(params, num_inputs + num_outputs, rnd);
            weights = alias_tensor(num_inputs, num_outputs);
            if (bias_mode == LINEAR_HAS_BIAS)
            {
                biases = alias_tensor(1, num_outputs);
                biases(params, weights.size()) = 0;
            }
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            const auto& prev_output = sub.get_output();
            DLIB_CASSERT((long)num_inputs == prev_output.nc(),
                "The number of input features to this linear layer doesn't match the size the linear layer was trained with");

            output.set_size(prev_output.num_samples(), prev_output.k(), prev_output.nr(), num_outputs);
            auto w = weights(params, 0);
            tt::c_gemm(0, output, 1, prev_output, false, w, false);            

            if (bias_mode == LINEAR_HAS_BIAS)
            {
                const auto b = biases(params, weights.size());
                const long output_plane_size = output.nr() * output.nc();
                for (long n = 0; n < output.num_samples(); ++n)
                {
                    for (long k = 0; k < output.k(); ++k)
                    {
                        auto output_slice = alias_tensor(output.nr(), output.nc())(output, (n * output.k() + k) * output_plane_size);
                        tt::add(1, output_slice, 1, b);
                    }
                }
            }
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& params_grad)
        {
            if (learning_rate_multiplier != 0) {
                auto pw = weights(params_grad, 0);
                tt::c_gemm(0, pw, learning_rate_multiplier, sub.get_output(), true, gradient_input, false);
                if (bias_mode == LINEAR_HAS_BIAS)
                {
                    auto pb = biases(params_grad, weights.size());
                    const long grad_plane_size = gradient_input.nr() * gradient_input.nc();
                    for (long n = 0; n < gradient_input.num_samples(); ++n)
                    {
                        for (long k = 0; k < gradient_input.k(); ++k)
                        {
                            auto grad_slice = alias_tensor(gradient_input.nr(), gradient_input.nc())(gradient_input, (n * gradient_input.k() + k) * grad_plane_size);
                            tt::assign_bias_gradient(pb, grad_slice);
                        }
                    }
                }
            }

            auto w = weights(params, 0);
            tt::c_gemm(1, sub.get_gradient_input(), 1, gradient_input, false, w, true);
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
                if (bias_mode_ != item.bias_mode) throw serialization_error("Wrong linear_bias_mode found while deserializing dlib::linear_");
                deserialize(item.learning_rate_multiplier, in);
            }
            else
            {
                throw serialization_error("Unexpected version '" + version + "' found while deserializing dlib::linear_.");
            }
        }

        friend std::ostream& operator<<(std::ostream& out, const linear_& item)
        {
            if (item.bias_mode == LINEAR_HAS_BIAS)
            {
                out << "linear (num_outputs=" << item.num_outputs << ")";
                out << " learning_rate_mult=" << item.learning_rate_multiplier;
            }
            else
            {
                out << "linear_no_bias (num_outputs=" << item.num_outputs << ")";
                out << " learning_rate_mult=" << item.learning_rate_multiplier;
            }
            return out;
        }

        friend void to_xml(const linear_& item, std::ostream& out)
        {
            if (item.bias_mode == LINEAR_HAS_BIAS)
            {
                out << "<linear"
                    << " num_outputs='" << item.num_outputs << "'"
                    << " learning_rate_mult='" << item.learning_rate_multiplier << "'>\n";
                out << mat(item.params);
                out << "</linear>\n";
            }
            else
            {
                out << "<linear_no_bias"
                    << " num_outputs='" << item.num_outputs << "'"
                    << " learning_rate_mult='" << item.learning_rate_multiplier << "'>\n";
                out << mat(item.params);
                out << "</linear_no_bias>\n";
            }
        }

    private:
        unsigned long num_inputs;
        unsigned long num_outputs;
        double learning_rate_multiplier;
        unsigned long bias_mode;
        resizable_tensor params;
        alias_tensor weights, biases;
    };

    template <unsigned long num_outputs, typename SUBNET>
    using linear = add_layer<linear_<num_outputs, LINEAR_HAS_BIAS>, SUBNET>;
    template <unsigned long num_outputs, typename SUBNET>
    using linear_no_bias = add_layer<linear_<num_outputs, LINEAR_NO_BIAS>, SUBNET>;

    // ----------------------------------------------------------------------------------------
    const long DEFAULT_NUM_HEADS = 4;

    template <long nb_heads>
    class hsplit_
    {
    public:
        hsplit_(long nb_heads_ = DEFAULT_NUM_HEADS) : num_heads(nb_heads_) {}

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
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            auto& grad = sub.get_gradient_input();

            tt::reorg_gradient2(true, grad, 1, num_heads, gradient_input);
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
        long num_heads;
    };

    template <long num_heads, typename SUBNET>
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
        }
        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            auto& grad = sub.get_gradient_input();

            tt::reorg2(true, grad, 1, grad.k(), gradient_input);
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
        tril_(): diag(diag_), diag_value(compute_diag_value()) {}
        
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

    template <template<typename> class tag>
    class multm_prev_ {
    public:
        const static unsigned long id = tag_id<tag>::id;

        multm_prev_() {}
        template <typename SUBNET> void setup(const SUBNET& /*sub*/) { }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output) {
            auto& t1 = sub.get_output();
            auto& t2 = layer<tag>(sub).subnet().get_output();
            output.set_size(t1.num_samples(), t1.k(), t1.nr(), t2.nc());

            tt::c_gemm(0, output, 1, t1, false, t2, false);
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/) {
            auto& t1 = sub.get_output();
            auto& t2 = layer<tag>(sub).get_output();
            auto& prev = sub.get_gradient_input();
            auto& prev_tag = layer<tag>(sub).get_gradient_input();            

            tt::c_gemm(1, prev, 1, gradient_input, false, t2, true);
            tt::c_gemm(1, prev_tag, 1, t1, true, gradient_input, false);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const multm_prev_& /*item*/, std::ostream& out) {
            serialize("multm_prev_", out);
        }
        friend void deserialize(multm_prev_& /*item*/, std::istream& in) {
            std::string version;
            deserialize(version, in);
            if (version != "multm_prev_")
                throw serialization_error("Unexpected version '" + version + "' found while deserializing dlib::multm_prev_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const multm_prev_& /*item*/) {
            out << "multm_prev" << id;
            return out;
        }
        friend void to_xml(const multm_prev_& /*item*/, std::ostream& out) {
            out << "<multm_prev tag='" << id << "'/>\n";
        }

    private:
        resizable_tensor params; // Not used
    };

    template <template<typename> class tag, typename SUBNET>
    using multm_prev = add_layer<multm_prev_<tag>, SUBNET>;

    template <typename SUBNET> using multm_prev1 = multm_prev<tag1, SUBNET>;
    template <typename SUBNET> using multm_prev2 = multm_prev<tag2, SUBNET>;
    template <typename SUBNET> using multm_prev3 = multm_prev<tag3, SUBNET>;
    template <typename SUBNET> using multm_prev4 = multm_prev<tag4, SUBNET>;
    template <typename SUBNET> using multm_prev5 = multm_prev<tag5, SUBNET>;
    using multm_prev1_ = multm_prev_<tag1>;
    using multm_prev2_ = multm_prev_<tag2>;
    using multm_prev3_ = multm_prev_<tag3>;
    using multm_prev4_ = multm_prev_<tag4>;
    using multm_prev5_ = multm_prev_<tag5>;

    template <unsigned long number_of_heads_, unsigned long embedding_dim_>
    class scale_weights_ : public multiply_ {
        static_assert(number_of_heads_ > 0, "The number of heads must be > 0");
        static_assert(embedding_dim_ > 0, "The embeddind size must be > 0");

    public:
        explicit scale_weights_() : multiply_(1.0f / std::sqrt(static_cast<float>(embedding_dim_ / number_of_heads_))) {}
    };
    template <unsigned long num_heads, unsigned long embedding_length, typename SUBNET>
    using scale_weights = add_layer<scale_weights_<num_heads, embedding_length>, SUBNET>;

    enum softmax_mode { CHANNEL_WISE = 0, PLANE_WISE = 1 };

    template <softmax_mode s_mode_>
    class softmax2_
    {
    public:
        softmax2_() {}

        template <typename SUBNET>
        void setup(const SUBNET& /*sub*/) {}

        void forward_inplace(const tensor& input, tensor& output)
        {
            tt::softmax2(output, input, s_mode_);
        }

        void backward_inplace(
            const tensor& computed_output,
            const tensor& gradient_input,
            tensor& data_grad,
            tensor& /*params_grad*/
        )
        {
            tt::softmax_gradient2(data_grad, computed_output, gradient_input, s_mode_);
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
            out << "softmax2 (mode=" << (s_mode_ == CHANNEL_WISE ? "channel_wise" : "plane_wise") << ")";
            return out;
        }

        friend void to_xml(const softmax2_& item, std::ostream& out)
        {
            out << "<softmax2 mode='" << (s_mode_ == CHANNEL_WISE ? "channel_wise" : "plane_wise") << "'/>\n";
        }

    private:
        resizable_tensor params; // unused
    };

    template <typename SUBNET>
    using softmax2 = add_layer<softmax2_<CHANNEL_WISE>, SUBNET>;

    template <typename SUBNET>
    using softmaxm = add_layer<softmax2_<PLANE_WISE>, SUBNET>;

    // Basic layers for Query, Key, and Value
    template <int nb_heads, int num_filters_out, typename SUBNET>
    //using query = linear_no_bias<num_filters_out, SUBNET>;
    using query = con<nb_heads, 1, 1, 1, nb_heads, SUBNET>;
    template <int nb_heads, int num_filters_out, typename SUBNET>
    //using key = linear_no_bias<num_filters_out, SUBNET>;
    using key = con<nb_heads, 1, 1, 1, nb_heads, SUBNET>;
    template <int nb_heads, int num_filters_out, typename SUBNET>
    //using value = linear_no_bias<num_filters_out, SUBNET>;
    using value = con<nb_heads, 1, 1, 1, nb_heads, SUBNET>;

    // Core masked multihead attention block
    template <int embedding_dim, int nb_heads, typename SUBNET>
    using multihead_attention_block =
        add_prev3<
        //linear<embedding_dim,
        cont<1, 1, nb_heads, 1, nb_heads,
        //hstack<
        multm_prev1<
        dropout_10<softmaxm<
        tril_mask<
        scale_weights<nb_heads, embedding_dim,
        multm_prev2<
        //hsplit<nb_heads, query<embedding_dim, skip3<
        query<nb_heads, embedding_dim, skip3<
        //tag2<transpose<hsplit<nb_heads, key<embedding_dim, skip3<
        tag2<transpose<key<nb_heads, embedding_dim, skip3<
        //tag1<hsplit<nb_heads, value<embedding_dim,
        tag1<value<nb_heads, embedding_dim,
        tag3<SUBNET>>>>>>>>>>>>>>>>>;
        //tag3<SUBNET>>>>>>>>>>>>>>>>>>>>>;

    // Feedforward blocks
    template <int N, typename SUBNET>
    using se = scale8<sig<fc<N, relu<bn_fc<fc<N / 16, avg_pool_everything<tag8<SUBNET>>>>>>>>;

    template <int sequence_dim, int embedding_dim, typename SUBNET>

    /*using feed_forward =
        add_prev5<
        scale5<con<1, 1, 1, 1, 1,
        //cont<1, sequence_dim, embedding_dim, sequence_dim, embedding_dim,
        sig<fc<embedding_size,
        dropout_10<gelu<bn_fc<fc<embedding_size / 4,
        avg_pool_everything<tag5<SUBNET>>>>>>>>>>>;*/
    using feed_forward =
        add_prev5<
        cont<1, sequence_dim, embedding_dim, sequence_dim, embedding_dim,
        fc<embedding_size,
        dropout_10<gelu<bn_fc<fc<embedding_size * 4,
        tag5<SUBNET>>>>>>>>;
    /*template <int embedding_dim, typename SUBNET>
    using feed_forward =
        add_prev5<
        linear<embedding_size,
        dropout_10<gelu<linear<embedding_size * 4,
        tag5<SUBNET>>>>>>;*/

    // Transformer block
    template <typename SUBNET>
    using transformer_block =
        feed_forward<sequence_size, embedding_size,
        multihead_attention_block<embedding_size, number_of_heads, 
        rms_norm<SUBNET>>>;

    // Classification head
    template <int num_logits, typename SUBNET>
    using classification_head = loss_multiclass_log<fc<num_logits, SUBNET>>;

    // VSLM network
    using llm_net = classification_head<vocab_size,        
        repeat<number_of_blocks, transformer_block,
        positional_embeddings<vocab_size, embedding_size,
        input<matrix<int, 0, 1>>>>>;
}

const size_t std_global_context_size = (5 * sequence_size);
class context_window {
public:
    context_window(size_t window_size, int pad_value = pad_id, size_t max_global_context_size = std_global_context_size)
        : window_size_(window_size), pad_value_(pad_value), max_global_context_size_(max_global_context_size),
        global_context_pos_(0), end_of_context_(true) {
        if (max_global_context_size < window_size) max_global_context_size = window_size;
    }

    void reset() {
        global_context_.clear();
        global_context_pos_ = 0;
        end_of_context_ = true;
    }

    void add_input(const std::vector<int>& input, bool reset_pos = false) {        
        global_context_.insert(global_context_.end(), input.begin(), input.end());
        end_of_context_ = false;
        if (global_context_.size() > max_global_context_size_) {
            size_t excess_size = global_context_.size() - max_global_context_size_;
            if (global_context_pos_ >= excess_size) {
                global_context_pos_ -= excess_size;
            } else {
                global_context_pos_ = 0;
            }
            global_context_.erase(global_context_.begin(), global_context_.begin() + excess_size);
        }
        if (reset_pos) global_context_pos_ = 0;
    }

    bool get_padded_window(matrix<int, 0, 1>& padded_window) {
        if (!end_of_context_) {
            std::vector<int> padded_window_(window_size_, pad_value_);
            size_t end_pos = std::min(window_size_, global_context_.size() - global_context_pos_);
            std::copy(global_context_.begin() + global_context_pos_, global_context_.begin() + global_context_pos_ + end_pos, padded_window_.begin());
            if (global_context_pos_ < global_context_.size()) global_context_pos_++;
            end_of_context_ = (global_context_pos_>= global_context_.size());
            padded_window = mat(padded_window_);
        }
        return !end_of_context_;
    }

    void add_output(int output_token, bool reset_pos = false) {
        std::vector<int> input;
        input.push_back(output_token);
        add_input(input, reset_pos);
    }

    bool is_end_of_context() const {
        return end_of_context_;
    }
    size_t get_global_context_size() const {
        return global_context_.size();
    }

private:
    std::vector<int> global_context_;
    size_t window_size_;
    int pad_value_;
    size_t max_global_context_size_;
    size_t global_context_pos_;
    bool end_of_context_;
};

class documents {
public:
    documents(size_t seq_size = sequence_size, int pad_value = pad_id, bool use_letter_tokenization = false) :
        sequence_size_(seq_size), pad_value_(pad_value), use_letter_tokenization_(use_letter_tokenization) {
        is_initialized_ = false;
        if (!use_letter_tokenization_) {
            if (fs::exists(vocabulary_prefix + ".model")) {
                auto status = sp_.Load(vocabulary_prefix + ".model");
                if (!status.ok()) cerr << "error loading SentencePiece model: " << status.ToString() << endl;
                else is_initialized_ = true;
            } else {
                cerr << "vocabulary file not found! (<" << (vocabulary_prefix + ".model|.vocab") << ">)" << endl;
            }
        } else {         
            is_initialized_ = true;
        }
        clear_all();
    }
    size_t get_total_tokens(void) { return total_tokens_; }
    void clear_all(void) {        
        source_tokens_.clear();
        pre_samples_.clear();
        pre_labels_.clear();
        total_tokens_ = 0;
        samples_idx_ = 0;
    }

    void load_text(const std::string& text, bool split_sentences) {
        if (!is_initialized_) return;        
        if (split_sentences) {
            std::vector<std::string> sentences = split_into_sentences(text);
            for (const auto& sentence : sentences) {
                std::vector<int> tokens = preprocess_sentence(sentence);
                if (tokens.size() != 0) {
                    source_tokens_.push_back(tokens);
                    total_tokens_ += tokens.size();
                }
            }
        } else {
            std::vector<int> tokens = preprocess_sentence(text);
            if (tokens.size() != 0) {
                source_tokens_.push_back(tokens);
                total_tokens_ += tokens.size();
            }
        }
        if (pre_samples_.size() > 0) {
            pre_samples_.clear();
            pre_labels_.clear();
            samples_idx_ = 0;
        }        
    }

    void load_documents(const std::string& path, bool split_sentences = true) {
        if (!is_initialized_) return;
        fs::path fs_path = fs::path(path);
        try {
            if (fs::is_regular_file(fs_path) && fs_path.extension() == ".txt") {
                cout << "loading file: " << fs_path.string() << endl;
                std::ifstream file(fs_path, std::ios::binary);
                std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
                if (!is_utf8(content)) cout << "warning - file <" << fs_path.string() << "> seems not to be UTF-8 encoded" << endl;
                load_text(content, split_sentences);
            } else if (fs::is_directory(fs_path)) {
                for (const auto& entry : fs::recursive_directory_iterator(fs_path)) {
                    if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                        cout << "loading file: " << entry.path().string() << endl;
                        std::ifstream file(entry.path(), std::ios::binary);
                        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
                        if (!is_utf8(content)) cout << "warning - file <" << entry.path().string() << "> seems not to be UTF-8 encoded" << endl;
                        load_text(content, split_sentences);
                    }
                }
            } else {
                cerr << "the specified path is neither a file nor a directory: " << fs_path.string() << endl;
            }
        } catch (const fs::filesystem_error& e) {
            cerr << "error accessing path: " << e.what() << endl;
        }
    }

    bool generate_samples(size_t num_samples, std::vector<matrix<int, 0, 1>>& samples, std::vector<unsigned long>& labels, bool select_randomly = true) {
        samples.clear();
        labels.clear();
        if (!is_initialized_ || source_tokens_.empty()) return false;

        if (select_randomly) {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::uniform_int_distribution<size_t> dist_sentence(0, source_tokens_.size() - 1);

            while (samples.size() < num_samples) {
                size_t sentence_idx = dist_sentence(gen);
                const std::vector<int>& sentence = source_tokens_[sentence_idx];

                std::uniform_int_distribution<size_t> dist_position(0, sentence.size() > (sequence_size_ + 1) ? sentence.size() - (sequence_size_ + 1) : sentence.size() - 1);
                size_t start_pos = dist_position(gen);
                int seq_length = std::min(sentence.size() - start_pos, sequence_size_ + 1);
                std::vector<int> input_tokens(sentence.begin() + start_pos, sentence.begin() + start_pos + seq_length);
                std::vector<int> sample(sequence_size_ + 1, pad_value_);
                std::copy(input_tokens.begin(), input_tokens.end(), sample.begin());
                unsigned long label = sample.back();
                sample.resize(sequence_size_);
                samples.push_back(mat(sample));
                labels.push_back(label);
            }
        } else {
            if (pre_samples_.size() == 0) {
                int i, j;
                for (const auto& sentence : source_tokens_) {                    
                    if (sentence.size() > (sequence_size_ + 1)) {                        
                        for (i = 0; i < (int)(sentence.size()) - (sequence_size_ + 1); ++i) {
                            matrix<int> sample(sequence_size_, 1);
                            for (j = 0; j < (int)sequence_size_; ++j) sample(j, 0) = sentence[i + j];
                            pre_samples_.push_back(sample);
                            pre_labels_.push_back(static_cast<unsigned long>(sentence[i + j]));
                        }
                    }                    
                }
            }
            if (pre_samples_.size() > 0) {
                while (samples.size() < num_samples) {
                    if (samples_idx_ >= pre_samples_.size()) samples_idx_ = 0;
                    samples.push_back(pre_samples_[samples_idx_]);
                    labels.push_back(pre_labels_[samples_idx_]);
                    samples_idx_++;
                }
            }
        }        
        return (samples.size() > 0);
    }

private:
    size_t sequence_size_;
    std::vector<std::vector<int>> source_tokens_;
    std::vector<matrix<int, 0, 1>> pre_samples_;
    std::vector<unsigned long> pre_labels_;
    size_t samples_idx_;

    sentencepiece::SentencePieceProcessor sp_;
    size_t total_tokens_;
    int pad_value_;
    bool use_letter_tokenization_;
    bool is_initialized_;

    std::vector<std::string> split_into_sentences(const std::string& text) {
        std::vector<std::string> sentences = dlib::split(text, "\r\n");
        return sentences;
    }

    std::vector<int> preprocess_sentence(const std::string& sentence, bool add_eos_id = false) {
        std::string cleaned_sentence = dlib::trim(replace_html_entities(std::regex_replace(sentence, std::regex("(.)\\1{4,}"), "$1$1$1$1")));
        std::vector<int> tokens;
        if (!use_letter_tokenization_) {
            sp_.Encode(cleaned_sentence, &tokens);
        } else {
            for (int i = 0; i < (int)(sentence.size()); ++i) tokens.push_back(static_cast<unsigned char>(sentence[i]));
        }        
        if (add_eos_id) tokens.push_back(eos_id);
        return tokens;
    }
};

void test_transpose()
{
    const long num_samples = 2;
    const long k = 3;
    const long nr = 4;
    const long nc = 5;

    resizable_tensor input(num_samples, k, nr, nc);
    resizable_tensor output_cpu_a(num_samples, k, nc, nr);
    tt::tensor_rand rnd(0);
    rnd.fill_uniform(input);
    resizable_tensor output_cpu_b(input);

    cpu::transpose(false, output_cpu_a, input);
    cpu::transpose(true, output_cpu_b, output_cpu_a);
    input *= 2;
    DLIB_TEST_MSG(max(abs(mat(output_cpu_b) - mat(input))) < 1e-5,
        "transpose_cpu: max(abs(mat(output_cpu_b) - mat(input))) < 1e-5");

#ifdef DLIB_USE_CUDA
    input /= 2;
    resizable_tensor output_cuda_a, output_cuda_b(input);
    output_cuda_a.copy_size(output_cpu_a);
    cuda::transpose(false, output_cuda_a, input);
    cuda::transpose(true, output_cuda_b, output_cuda_a);
    DLIB_TEST_MSG(max(abs(mat(output_cpu_a) - mat(output_cuda_a))) < 1e-5,
        "transpose_cuda: max(abs(mat(output_cpu_a) - mat(output_cuda_a))) < 1e-5");
    DLIB_TEST_MSG(max(abs(mat(output_cpu_b) - mat(output_cuda_b))) < 1e-5,
        "transpose_cuda: max(abs(mat(output_cpu_b) - mat(output_cuda_b))) < 1e-5");
#endif
}

void test_hsplit_hstack() {
    const long num_heads = 4;
    const long num_samples = 1;
    const long input_k = 1;
    const long input_nr = 8;
    const long input_nc = 12;

    using net_type = tag1<hstack<hsplit<num_heads, input<matrix<float>>>>>;
    net_type net;

    resizable_tensor input_tensor;
    input_tensor.set_size(num_samples, input_k, input_nr, input_nc);
    tt::tensor_rand rnd;
    rnd.fill_uniform(input_tensor);

    net.forward(input_tensor);
    auto& output_tensor = layer<tag1>(net).get_output();

    DLIB_TEST_MSG(output_tensor.num_samples() == input_tensor.num_samples(),
        "hsplit_hstack: output_tensor.num_samples() == input_tensor.num_samples()");
    DLIB_TEST_MSG(output_tensor.k() == input_tensor.k(),
        "hsplit_hstack: output_tensor.k() == input_tensor.k()");
    DLIB_TEST_MSG(output_tensor.nr() == input_tensor.nr(),
        "hsplit_hstack: output_tensor.nr() == input_tensor.nr()");
    DLIB_TEST_MSG(output_tensor.nc() == input_tensor.nc(),
        "hsplit_hstack: output_tensor.nc() == input_tensor.nc()");
    DLIB_TEST_MSG(max(abs(mat(output_tensor) - mat(input_tensor))) < 1e-5,
        "hsplit_hstack: max(abs(mat(output_tensor) - mat(input_tensor))) < 1e-5");

    /*const long num_samples = 1;
    const long num_channels = 1;
    const long num_rows = 4;
    const long num_cols = 6;

    resizable_tensor input;
    input.set_size(num_samples, num_channels, num_rows, num_cols);
    tt::tensor_rand rnd(0);
    rnd.fill_uniform(input);

    const int row_stride = 1;
    const int col_stride = 2;
    const long output_channels = num_channels * row_stride * col_stride;
    const long output_rows = num_rows / row_stride;
    const long output_cols = num_cols / col_stride;

    resizable_tensor output_cpu, output_cpu2;
    output_cpu.set_size(num_samples, output_channels, output_rows, output_cols);
    output_cpu2.set_size(num_samples, output_channels, output_rows, output_cols);

#ifdef DLIB_USE_CUDA
    resizable_tensor output_cuda;
    output_cuda.set_size(num_samples, output_channels, output_rows, output_cols);
#endif

    cpu::reorg(output_cpu, row_stride, col_stride, input);
    cpu::reorg2(false, output_cpu2, row_stride, col_stride, input);
    DBG_INFO("reorg_input: ", input, true);
    DBG_INFO("reorg_output: ", output_cpu, true);
    DBG_INFO("reorg2_output: ", output_cpu2, true);

    resizable_tensor grad_cpu;
    grad_cpu.copy_size(input);
    cpu::reorg_gradient2(false, grad_cpu, row_stride, col_stride, output_cpu2);
    DBG_INFO("reorg_gradient2_output: ", grad_cpu, true);
    DLIB_TEST_MSG(max(abs(mat(grad_cpu) - mat(input))) < 1e-5,
        "reorg_cpu: max(abs(mat(grad_cpu) - mat(input))) < 1e-5");*/
}

/*    const long num_heads = 2;
    input.set_size(2, 1, 4, 6);
    rnd.fill_uniform(input);
    resizable_tensor output, input2;
    output.set_size(input.num_samples(), input.k() * num_heads,
        input.nr(), input.nc() / num_heads);
    input2.copy_size(input);

#ifdef DLIB_USE_CUDA
    cuda::split_columns(false, output, input, num_heads);
    DBG_INFO("split_src: ", input, true);
    DBG_INFO("split_dst: ", output, true);
    cuda::merge_columns(false, input2, output);
    DBG_INFO("merge_dst: ", input2, true);
#endif
}*/

void test_positional_encodings()
{
    using net_type = tag1<positional_encodings<input<matrix<float>>>>;
    net_type net;

    const unsigned long sequence_dim = 4;
    const unsigned long embedding_dim = 6;
    const unsigned long n_samples = 1, n_channels = 1;
    matrix<float> input_data(sequence_dim, embedding_dim);
    input_data = 0.0f;
    
    resizable_tensor input_tensor(n_samples, n_channels, sequence_dim, embedding_dim);
    std::vector<matrix<float>> x(n_samples);
    x[0] = input_data;
    net.to_tensor(&x[0], &x[0] + n_samples, input_tensor);
    net.forward(input_tensor);

    matrix<float> expected_output(sequence_dim, embedding_dim);
    const float n = 10000.0f;
    for (long r = 0; r < sequence_dim; ++r) {
        for (long c = 0; c < embedding_dim; ++c) {
            float theta = static_cast<float>(r) / std::pow(n, static_cast<float>(c) / embedding_dim);
            expected_output(r, c) = (c % 2 == 0) ? std::sin(theta) : std::cos(theta);
        }
    }    

    auto& net_output = layer<tag1>(net).get_output();
    DBG_INFO("net_output: ", net_output, true);
    cout << "expected_output: " << expected_output << endl;
    DLIB_TEST_MSG(max(abs(mat(net_output) - expected_output)) < 1e-5, "positional_encodings layer");
}

void test_embeddings()
{
    const size_t num_sequences = 200, sequence_length = 7, num_classes = 10, num_tokens = 100, embedding_length = 5;
    using net_type = loss_multiclass_log<fc<num_classes,
        relu<fc<16,relu<fc<32,
        embeddings<num_tokens, embedding_length,
        input<matrix<unsigned long, 0, 1>>>>>>>>>;
    net_type net;
    dnn_trainer<net_type> trainer(net, sgd(0, 0.9));
    trainer.set_learning_rate(1e-1);
    trainer.set_min_learning_rate(1e-4);
    trainer.set_mini_batch_size(16);
    trainer.set_max_num_epochs(200);

    dlib::rand rnd;
    auto generate_sequences = [&](size_t num_sequences, size_t sequence_length, size_t num_tokens) {
        std::vector<matrix<unsigned long, 0, 1>> sequences;
        for (size_t i = 0; i < num_sequences; ++i)
        {
            matrix<unsigned long, 0, 1> seq(sequence_length, 1);
            for (size_t j = 0; j < sequence_length; ++j)
                seq(j, 0) = rnd.get_random_32bit_number() % num_tokens;
            sequences.push_back(seq);
        }
        return sequences;
    };

    auto generate_labels = [&](size_t num_sequences, size_t num_classes) {
        std::vector<unsigned long> labels;
        for (size_t i = 0; i < num_sequences; ++i)
            labels.push_back(rnd.get_random_32bit_number() % num_classes);
        return labels;
    };

    auto sequences = generate_sequences(num_sequences, sequence_length, num_tokens);
    auto labels = generate_labels(num_sequences, num_classes);

    trainer.train(sequences, labels);
    std::vector<unsigned long> predicted_labels = net(sequences);
    size_t num_correct = 0;
    for (size_t i = 0; i < labels.size(); ++i)
    {
        if (predicted_labels[i] == labels[i]) ++num_correct;
    }
    double acc = static_cast<double>(num_correct) / labels.size();
    DLIB_TEST_MSG(acc > 0.99, "embeddings accuracy: " + to_string(acc));
}

void test_rms_normalize()
{
    resizable_tensor x(2, 3, 4, 5);
    resizable_tensor y_cpu(x);
    tt::tensor_rand rnd(0);
    rnd.fill_uniform(x);
    resizable_tensor scale_cpu;
    resizable_tensor gamma(1, x.k());
    gamma = 1;
    const float eps = 1e-5;
    cpu::rms_normalize(eps, y_cpu, scale_cpu, x, gamma);

    // Check that the output is correctly normalized
    const float* p_x = x.host();
    const float* p_y = y_cpu.host();
    const float* p_scale = scale_cpu.host();
    bool error_found = false;
    for (long n = 0; n < x.num_samples(); ++n)
    {
        for (long k = 0; k < x.k(); ++k)
        {
            for (long r = 0; r < x.nr(); ++r)
            {
                for (long c = 0; c < x.nc(); ++c)
                {
                    float x_val = p_x[tensor_index(x, n, k, r, c)];
                    float y_val = p_y[tensor_index(y_cpu, n, k, r, c)];
                    float rms_val = p_scale[n];
                    if (std::abs(y_val - x_val * rms_val) >= 1e-5) error_found = true;
                }
            }
        }
    }
    DLIB_TEST_MSG(!error_found, "Normalized values vs expected values");

    // Check the backward pass
    resizable_tensor gradient_input(x);
    resizable_tensor src_grad_cpu(x), gamma_grad_cpu(1, x.k());
    resizable_tensor dscale_cpu(x.num_samples());
    rnd.fill_gaussian(gradient_input);
    src_grad_cpu = 0;
    cpu::rms_normalize_gradient(gradient_input, scale_cpu, x, gamma, src_grad_cpu, gamma_grad_cpu, dscale_cpu);

    const float* p_gradient_input = gradient_input.host();
    const float* p_src = x.host();
    const float* p_src_grad_cpu = src_grad_cpu.host();
    const float* p_gamma = gamma.host();
    const float* p_scale_cpu = scale_cpu.host();
    const float* p_dscale_cpu = dscale_cpu.host();

    bool backward_error_found = false;
    for (long n = 0; n < x.num_samples(); ++n)
    {
        const float scale_pow = -0.5 * std::pow(p_scale_cpu[n], 3.0f);
        for (long k = 0; k < x.k(); ++k)
        {
            for (long r = 0; r < x.nr(); ++r)
            {
                for (long c = 0; c < x.nc(); ++c)
                {
                    float gradient_input_val = p_gradient_input[tensor_index(gradient_input, n, k, r, c)];
                    float src_val = p_src[tensor_index(x, n, k, r, c)];
                    float rms_val = p_scale_cpu[n];
                    float expected_src_grad = gradient_input_val * p_gamma[k] * rms_val + p_dscale_cpu[n] * 2 * src_val * 1.0f / (x.k() * x.nr() * x.nc());
                    float src_grad_val = p_src_grad_cpu[tensor_index(src_grad_cpu, n, k, r, c)];
                    if (std::abs(src_grad_val - expected_src_grad) >= 1e-5) backward_error_found = true;
                }
            }
        }
    }
    DLIB_TEST_MSG(!backward_error_found, "Backward pass values vs expected values");

#ifdef DLIB_USE_CUDA
    resizable_tensor y_cuda(x);
    resizable_tensor scale_cuda;
    cuda::rms_normalize(eps, y_cuda, scale_cuda, x, gamma);
    DLIB_TEST_MSG(max(abs(mat(y_cpu) - mat(y_cuda))) < 1e-5, "max(abs(mat(y_cpu) - mat(y_cuda))) < 1e-5");
    DLIB_TEST_MSG(max(abs(mat(scale_cpu) - mat(scale_cuda))) < 1e-5,
        "max(abs(mat(scale_cpu) - mat(scale_cuda))) < 1e-5");

    resizable_tensor src_grad_cuda(x), gamma_grad_cuda(1, x.k());
    resizable_tensor dscale_cuda(x.num_samples());
    src_grad_cuda = 0;
    cuda::rms_normalize_gradient(gradient_input, scale_cuda, x, gamma, src_grad_cuda, gamma_grad_cuda, dscale_cuda);
    DLIB_TEST_MSG(max(abs(mat(src_grad_cpu) - mat(src_grad_cuda))) < 1e-5,
        "max(abs(mat(src_grad_cpu) - mat(src_grad_cuda))) < 1e-5");
    DLIB_TEST_MSG(max(abs(mat(gamma_grad_cpu) - mat(gamma_grad_cuda))) < 1e-5,
        "max(abs(mat(gamma_grad_cpu) - mat(gamma_grad_cuda))) < 1e-5");
    DLIB_TEST_MSG(max(abs(mat(dscale_cpu) - mat(dscale_cuda))) < 1e-5,
        "max(abs(mat(dscale_cpu) - mat(dscale_cuda))) < 1e-5");
#endif
}

void test_tril()
{   
    using net_type = tag1<tril_mask<tag2<input<matrix<float>>>>>;
    net_type net;

    // Input tensor
    dlib::rand rnd;
    const int nr = 2, nc = 3;
    const int n_samples = 3, k = 1;
    std::vector<matrix<float>> x(n_samples);
    matrix<float> xtmp(nr, nc);
    for (int ii = 0; ii < n_samples; ++ii) {
        for (int jj = 0; jj < nr; ++jj)
            for (int kk = 0; kk < nc; ++kk)
                xtmp(jj, kk) = rnd.get_random_gaussian();
        x[ii] = xtmp;
    }

    // Convert input matrix to tensor
    resizable_tensor input_tensor;
    net.to_tensor(&x[0], &x[0] + n_samples, input_tensor);
    net.forward(input_tensor);

    // Expected output tensor (manually set for comparison)
    resizable_tensor expected_output;
    expected_output.copy_size(input_tensor);
    tt::copy_tensor(false, expected_output, 0, input_tensor, 0, input_tensor.k());
    for (int ii = 0; ii < n_samples; ++ii) {
        expected_output.host()[tensor_index(expected_output, ii, 0, 0, 1)] = -std::numeric_limits<float>::infinity();
        expected_output.host()[tensor_index(expected_output, ii, 0, 0, 2)] = -std::numeric_limits<float>::infinity();
        expected_output.host()[tensor_index(expected_output, ii, 0, 1, 2)] = -std::numeric_limits<float>::infinity();
    }

    // Compare output tensor with expected output
    auto& net_output = layer<tag1>(net).get_output();
    DLIB_TEST_MSG(max(abs(mat(net_output) - mat(expected_output))) < 1e-5, "tril layer");
}

int main(int argc, char* argv[]) {
    string corpus_dir;
    bool do_benchmark = false, text_generation = false;
    bool voc_training = false, model_training = false, model_prompting = false, use_sync_file = false;
    double learning_rate = 1e-3, min_learning_rate = 1e-6, weight_decay = 0.001, beta1 = 0.9, beta2 = 0.999, temperature = 0.9;
    long mini_batch_size = 16, iterations_without_progress_threshold = 50000, top_k = 3;
    std::vector<int> gpus = { 0 };
    set_dnn_prefer_fastest_algorithms();
       
    configure_console();
    cout << endl <<
        "███████╗██████╗ ███╗   ██╗██╗███████╗\n"
        "██╔════╝██╔══██╗████╗  ██║██║██╔════╝\n"
        "█████╗  ██████╔╝██╔██╗ ██║██║█████╗  \n"
        "██╔══╝  ██╔══██╗██║╚██╗██║██║██╔══╝  \n"
        "███████╗██║  ██║██║ ╚████║██║███████╗\n"
        "╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚══════╝\n"
        "Welcome to the ERNIE generative AI program! (version 1.1.4)\n\n";
    try {
        string msg_learning_rate = string("Initial learning rate for training (") + std::format("{:g}", learning_rate) + string(")"),
            msg_min_learning_rate = string("Minimum learning rate (") + std::format("{:g}", min_learning_rate) + string(")"),
            msg_iterations_without_progress_threshold = string("Iterations without progress threshold (") + std::format("{:0}", iterations_without_progress_threshold) + string(")"),
            msg_mini_batch_size = string("Mini batch size for training (") + std::format("{:0}", mini_batch_size) + string(")"),
            msg_weight_decay = string("Weight decay value for the Adam solver (") + std::format("{}", weight_decay) + string(")"),
            msg_beta1 = string("beta_1 parameter for the Adam solver (") + std::format("{}", beta1) + string(")"),
            msg_beta2 = string("beta_2 parameter for the Adam solver (") + std::format("{}", beta2) + string(")"),
            msg_temperature = string("Temperature for text generation (") + std::format("{:0}", temperature) + string(")"),
            msg_top_k = string("Top K for text generation (") + std::format("{:0}", top_k) + string(")");
        po::options_description desc("Options");
        desc.add_options()
            ("help,h", "Display help")
            ("corpus-directory,d", po::value<std::string>(&corpus_dir), "Directory containing text files to process")
            ("tokenize-sentences,s", po::bool_switch(&voc_training), "Tokenize static sentences to train vocabulary")
            ("train-model,m", po::bool_switch(&model_training), "Train a new language model")
            ("chat-mode,c", po::bool_switch(&model_prompting), "Use a trained language model to generate text")
            ("learning-rate,l", po::value<double>(&learning_rate), msg_learning_rate.c_str())
            ("min-learning-rate,n", po::value<double>(&min_learning_rate), msg_min_learning_rate.c_str())
            ("iter-without-progress,i", po::value<long>(&iterations_without_progress_threshold), msg_iterations_without_progress_threshold.c_str())
            ("mini-batch-size,b", po::value<long>(&mini_batch_size), msg_mini_batch_size.c_str())
            ("weight-decay,w", po::value<double>(&weight_decay), msg_weight_decay.c_str())
            ("beta-1", po::value<double>(&beta1), msg_beta1.c_str())
            ("beta-2", po::value<double>(&beta2), msg_beta2.c_str())
            ("gpus,g", po::value<std::vector<int>>(&gpus)->multitoken(), "List of GPU indices to use, e.g., --gpus 0 1 2")
            ("use-sync,y", po::bool_switch(&use_sync_file), "Enable a synchronization file during training")
            ("temperature,t", po::value<double>(&temperature), msg_temperature.c_str())
            ("top-k,k", po::value<long>(&top_k), msg_top_k.c_str())
            ("text-generation,o", po::bool_switch(&text_generation), "Generate text using the current model")
            ("benchmark", po::bool_switch(&do_benchmark), "Do unit tests of the Dlib functions");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::ostringstream oss;
            oss << desc;
            cout << oss.str() << endl;
            return 0;
        }
        po::notify(vm);
    } catch (const po::error& e) {
        cerr << "error: " << e.what() << endl;
        return 1;
    }
    // Set the signal handler for SIGINT
    signal(SIGINT, signalHandler);
    
    sentencepiece::SentencePieceProcessor sp;
    sentencepiece::util::Status status;
    if (do_benchmark) {
        const bool display_debug_info = false;
        const bool skip_tests[] = {
            true,      // 0: strings & tokenization
            true,      // 1: transpose layer
            true,      // 2: tril layer
            true,      // 3: positional_encodings layer
            false,      // 4: embeddings layer
            true,      // 5: softmax layer
            true,      // 6: attention mechanism
            true,      // 7: linear layer            
            true,      // 8: hsplit/hstack layers
            true,      // 9: rms_norm layer
            true,      // 10: multihead attention model
            false       // 11: "shakespeare" example
        };

        // test: tokenization
        if (!skip_tests[0]) {
            if (display_debug_info) cout << "test: strings & tokenization\n";
            std::string sentence = "  &nbsp;&lt;p&gt;Hellooooooo     frieeeends !!!!!! This is sooooooo coooool &amp; awesoooooome !&lt;/p&gt;  ";
            std::string cleaned_sentence = dlib::trim(replace_html_entities(std::regex_replace(sentence, std::regex("(.)\\1{4,}"), "$1$1$1$1")));
            cout << "string normalisation: [" << sentence << "] => [" << cleaned_sentence << "]" << endl;

            if (fs::exists(vocabulary_prefix + ".model")) status = sp.Load(vocabulary_prefix + ".model");
            else {
                cerr << "vocabulary file not found! (<" << (vocabulary_prefix + ".model|.vocab") << ">)" << endl;
            }
            std::vector<std::string> test_sentences = {
                "This is a test sentence in English.",
                "Ceci est une phrase de test en français.</s>",
                "Dies ist ein Testsatz auf Deutsch.",
                "<s>Questa è una frase di prova in italiano.</s>",
                "Esta es una frase de <unk> en español."
            };
            for (const auto& sentence : test_sentences) {
                std::vector<std::string> tokens;
                sp.Encode(sentence, &tokens);
                cout << "sentence: " << sentence << endl << "Tokens: ";
                for (const auto& token : tokens) cout << token << " ";

                std::vector<int> ids;
                sp.Encode(sentence, &ids);
                cout << endl << "token IDs: ";
                for (const auto& id : ids) cout << id << " ";

                string recovered_text;
                sp.Decode(ids, &recovered_text);
                cout << endl << "original sentence : " << recovered_text << endl << endl;
            }
        }

        // test: transpose layer
        if (!skip_tests[1]) {
            if (display_debug_info) cout << "test: transpose layer\n";
            test_transpose();
            {
                transpose_ l;
                auto res = test_layer(l);
                DLIB_TEST_MSG(res, " transpose test_0 layer\n" + res);
            }
        }        

        // test: tril layer
        if (!skip_tests[2]) {
            if (display_debug_info) cout << "\ntest: tril layer\n";
            test_tril();
            {
                tril_<0, neg_infinity_tag> l;
                auto res = test_layer(l);
                DLIB_TEST_MSG(res, " tril test_0 layer\n" + res);
            }
            {
                tril_<3, zero_tag> l;
                auto res = test_layer(l);
                DLIB_TEST_MSG(res, " tril test_1 layer\n" + res);
            }
            {
                tril_<-5, void, 1, 2> l;
                auto res = test_layer(l);
                DLIB_TEST_MSG(res, " tril test_2 layer\n" + res);
            }
        }

        // test: positional_encodings layer
        if (!skip_tests[3]) {
            if (display_debug_info) cout << "test: positional_encodings layer\n";
            test_positional_encodings();
            {
                positional_encodings_ l;
                auto res = test_layer(l);
                DLIB_TEST_MSG(res, " positional_encodings test layer\n" + res);
            }
        }

        // test: embeddings layer
        if (!skip_tests[4]) {
            if (display_debug_info) cout << "test: embeddings layer\n";
            test_embeddings();
            {
                embeddings_<7, 12> l;
                auto res = test_layer(l);
                DLIB_TEST_MSG(res, " embeddings test layer\n" + res);
            }
        }

        // test: softmax layer
        if (!skip_tests[5]) {
            if (display_debug_info) cout << "\ntest: softmax layer\n";
            {
                using net_type = tag1<softmaxm<tag2<input<matrix<float>>>>>;
                net_type net;

                // Input tensor
                const float NEG_INF = -std::numeric_limits<float>::infinity();
                dlib::rand rnd;
                const long nr = 2, nc = 3;
                const int n_samples = 3, k = 1;
                std::vector<matrix<float>> x(n_samples);
                matrix<float> xtmp(nr, nc);
                for (int ii = 0; ii < n_samples; ++ii) {
                    for (int jj = 0; jj < nr; ++jj)
                        for (int kk = 0; kk < nc; ++kk) {
                            float r = rnd.get_random_gaussian();
                            if (r > 1 || r < -1) r = NEG_INF;
                            xtmp(jj, kk) = r;
                        }
                    x[ii] = xtmp;
                }

                // Convert input matrix to tensor
                resizable_tensor input_tensor;
                net.to_tensor(&x[0], &x[0] + n_samples, input_tensor);
                net.forward(input_tensor);
                if (display_debug_info) DBG_INFO("input_tensor: ", input_tensor, true);

                // Expected output tensor
                resizable_tensor expected_output;
                expected_output.copy_size(input_tensor);
                for (int ii = 0; ii < n_samples; ++ii) {
                    for (int jj = 0; jj < nr; ++jj) {
                        matrix<float> m(1, nc);
                        bool all_neg_inf = true;
                        for (int kk = 0; kk < nc; ++kk) {
                            m(0, kk) = input_tensor.host()[tensor_index(input_tensor, ii, 0, jj, kk)];
                            if (m(0, kk) > NEG_INF) all_neg_inf = false;
                        }

                        matrix<float> r(1, nc);
                        if (all_neg_inf) {
                            for (int kk = 0; kk < nc; ++kk) r(0, kk) = (1.0f / nc);
                        }
                        else {
                            // Stabilize the computation by subtracting the max value
                            float max_val = max(m);
                            matrix<float> exp_m = exp(m - max_val);
                            float sum_exp = sum(exp_m) + std::numeric_limits<float>::epsilon();
                            r = exp_m / sum_exp;
                        }
                        for (int kk = 0; kk < nc; ++kk) {
                            expected_output.host()[tensor_index(expected_output, ii, 0, jj, kk)] = r(0, kk);
                        }
                    }
                }
                if (display_debug_info) DBG_INFO("expected_output: ", expected_output, true);

                // Compare output tensor with expected output
                auto& net_output = layer<tag1>(net).get_output();
                if (display_debug_info) DBG_INFO("net_output: ", net_output, true);
                DLIB_TEST_MSG(max(abs(mat(net_output) - mat(expected_output))) < 1e-5, "softmaxm layer");
            }
        }

        // test: attention mechanism
        if (!skip_tests[6]) {
            if (display_debug_info) cout << "\ntest: attention mechanism\n";
            {
                matrix<float> X(4, 3), WQ(3, 3), WK(3, 3), WV(3, 3);
                X = 1, 0, 0,
                    0, 1, 0,
                    1, 1, 0,
                    0, 0, 1;
                WQ = 2, 0, 2,
                    2, 0, 0,
                    2, 1, 2;
                WK = 2, 2, 2,
                    0, 2, 1,
                    0, 1, 1;
                WV = 1, 1, 0,
                    0, 1, 1,
                    0, 0, 0;

                // Calculate the matrices Q, K, V
                matrix<float> Q = X * WQ;
                matrix<float> K = X * WK;
                matrix<float> V = X * WV;

                // Calculate the attention scores
                auto local_softmax = [](const matrix<float>& m) {
                    matrix<float> result(m.nr(), m.nc());
                    for (long r = 0; r < m.nr(); ++r) {
                        bool all_neg_inf = true;
                        for (long c = 0; c < m.nc(); ++c) {
                            if (m(r, c) > -std::numeric_limits<float>::infinity()) {
                                all_neg_inf = false;
                                break;
                            }
                        }
                        if (all_neg_inf) {
                            for (long c = 0; c < m.nc(); ++c) result(r, c) = (1.0f / m.nc());
                        }
                        else {
                            float max_val = max(rowm(m, r));
                            matrix<float> exp_row = exp(rowm(m, r) - max_val);
                            float sum_exp = sum(exp_row) + std::numeric_limits<float>::epsilon();
                            for (long c = 0; c < m.nc(); ++c) {
                                result(r, c) = exp_row(0, c) / sum_exp;
                            }
                        }
                    }
                    return result;
                    };
                matrix<float> scores = Q * trans(K);
                matrix<float> attention_weights = local_softmax(scores / sqrt(static_cast<float>(K.nc())));

                // Calculate the output Z
                matrix<float> Z = attention_weights * V;

                // Display theoretical results
                if (display_debug_info) {
                    std::cout << "Q:\n" << Q << std::endl;
                    std::cout << "K:\n" << K << std::endl;
                    std::cout << "V:\n" << V << std::endl;
                    std::cout << "scores:\n" << scores << std::endl;
                    std::cout << "attention weights (softmax):\n" << attention_weights << std::endl;
                    std::cout << "Z:\n" << Z << std::endl;
                }

                // Model definition
                using net_type = tag10<multm_prev1<tag7<softmaxm<scale_weights<1, 3, tag6<multm_prev4<
                    tag3<linear<3, // Q
                    skip5<tag4<transpose<tag2<linear<3, // K
                    skip5<tag1<linear<3, // V
                    tag5<input<matrix<float>>>>>>>>>>>>>>>>>>>>;
                net_type net;

                // Convert X into a tensor
                const long nr = X.nr(), nc = X.nc();
                const int n_samples = 1, k = 1;
                std::vector<matrix<float>> xx(n_samples);
                matrix<float> xtmp(nr, nc);
                for (int ii = 0; ii < n_samples; ++ii) xx[ii] = X;
                resizable_tensor input_tensor;
                net.to_tensor(&xx[0], &xx[0] + n_samples, input_tensor);
                net.forward(input_tensor);

                // Initialise network weights
                for (long r = 0; r < WV.nr(); ++r) {
                    for (long c = 0; c < WV.nc(); ++c) {
                        layer<tag1>(net).subnet().layer_details().get_layer_params().host()[tensor_index(layer<tag1>(net).subnet().layer_details().get_layer_params(), r, c, 0, 0)] = WV(r, c);
                        layer<tag2>(net).subnet().layer_details().get_layer_params().host()[tensor_index(layer<tag2>(net).subnet().layer_details().get_layer_params(), r, c, 0, 0)] = WK(r, c);
                        layer<tag3>(net).subnet().layer_details().get_layer_params().host()[tensor_index(layer<tag3>(net).subnet().layer_details().get_layer_params(), r, c, 0, 0)] = WQ(r, c);
                    }
                }
                // Forward X again through the network
                net.forward(input_tensor);

                // Display network outputs
                auto& net_Q = layer<tag3>(net).get_output();
                auto& net_K = layer<tag2>(net).get_output();
                auto& net_V = layer<tag1>(net).get_output();
                auto& net_S = layer<tag6>(net).get_output();
                auto& net_AW = layer<tag7>(net).get_output();
                if (display_debug_info) {
                    DBG_INFO("net_Q (Q): ", net_Q, true);
                    DBG_INFO("net_K (K): ", net_K, true);
                    DBG_INFO("net_V (V): ", net_V, true);
                    DBG_INFO("net_S (scores): ", net_S, true);
                    DBG_INFO("net_AW (softmax): ", net_AW, true);
                }

                // Compare output tensor with expected output
                auto& net_output = layer<tag10>(net).get_output();
                if (display_debug_info) DBG_INFO("net_output (Z): ", net_output, true);
                DLIB_TEST_MSG(max(abs(mat(net_output) - Z)) < 1e-5, "attention mechanism");
            }
        }

        // test: linear layer
        if (!skip_tests[7]) {
            if (display_debug_info) cout << "\ntest: linear layer\n";
            {
                using net_type = tag1<linear<5, tag2<input<matrix<float>>>>>;
                net_type net;

                // Input tensor
                resizable_tensor input_tensor;
                input_tensor.set_size(1, 1, 2, 3);
                tt::tensor_rand rnd;
                rnd.fill_gaussian(input_tensor);

                // Convert input matrix to tensor
                net.forward(input_tensor);

                // Expected output tensor (manually set for comparison)
                resizable_tensor expected_output;
                expected_output.set_size(1, 1, 2, 5);
                auto w = mat(layer<tag1>(net).subnet().layer_details().get_weights());
                for (long i = 0; i < input_tensor.nr(); ++i) {
                    for (long j = 0; j < input_tensor.nc(); ++j) {
                        for (long k = 0; k < w.nc(); ++k) {
                            float val = 0.0f;
                            for (long l = 0; l < w.nr(); ++l) val += input_tensor.host()[tensor_index(input_tensor, 0, 0, i, l)] * w(l, k);
                            expected_output.host()[tensor_index(expected_output, 0, 0, i, k)] = val;
                        }
                    }
                }
                // Compare output tensor with expected output
                auto& net_ouput = layer<tag1>(net).get_output();
                DLIB_TEST_MSG(max(abs(mat(net_ouput) - mat(expected_output))) < 1e-5, "linear layer");
            }
            {
                linear_<1, LINEAR_NO_BIAS> l;
                auto res = test_layer(l);
                DLIB_TEST_MSG(res, " linear test_0 layer\n" + res);
            }
            {
                linear_<5, LINEAR_HAS_BIAS> l;
                auto res = test_layer(l);
                DLIB_TEST_MSG(res, " linear test_1 layer\n" + res);
            }
            {
                linear_<4, LINEAR_NO_BIAS> l;
                auto res = test_layer(l);
                DLIB_TEST_MSG(res, " linear test_2 layer\n" + res);
            }
        }

        /*
        if (!skip_tests[6]) {
            if (display_debug_info) cout << "\ntest: add_prev1 layer\n";
            {
                // Define the network
                using net_type = tag3<add_prev1<tag2<linear_no_bias<4, tag1<input<matrix<float>>>>>>>;
                net_type net;

                // Input tensor
                const int n_samples = 1, k = 1;
                std::vector<matrix<float>> x(n_samples);
                matrix<float> xtmp(2, 4);
                xtmp = 1.0f, 2.0f, 3.0f, 4.0f,
                       5.0f, 6.0f, 7.0f, 8.0f;
                x[0] = xtmp;

                // Convert input matrix to tensor
                resizable_tensor input_tensor;
                net.to_tensor(&x[0], &x[0] + n_samples, input_tensor);
                net.forward(input_tensor);

                // Get the internal linear weights
                matrix<float> w = mat(layer<tag2>(net).subnet().layer_details().get_weights());

                // Theoretical calculation of the output
                matrix<float> input_matrix = x[0];
                matrix<float> linear_output = input_matrix * w;
                matrix<float> expected_output = linear_output + input_matrix;

                // Compare output tensor with expected output
                auto& net_output = layer<tag3>(net).get_output();

                // Display results
                if (display_debug_info) {
                    cout << "input matrix:" << input_matrix << endl;
                    cout << "weights matrix:" << w << endl;
                    cout << "expected output matrix:" << expected_output << endl;
                    DBG_INFO("network output matrix: ", net_output, true);
                }
                DLIB_TEST_MSG(max(abs(mat(net_output) - expected_output)) < 1e-5, "add_prev1 layer");
            }
        }
        */

        // test: hsplit/hstack layers
        if (!skip_tests[8]) {
            if (display_debug_info) cout << "\ntest: hsplit/hstack layers\n";                            
            test_hsplit_hstack();
            {
                hstack_ l;
                auto res = test_layer(l);
                DLIB_TEST_MSG(res, " hsplit/hstack layers\n" + res);
            }
        }

        // test: rms_norm layer
        if (!skip_tests[9]) {
            if (display_debug_info) cout << "\ntest: rms_norm layer\n";
            {
                test_rms_normalize();
                {
                    rms_norm_ l;
                    auto res = test_layer(l);
                    DLIB_TEST_MSG(res, " RMS normalize layer" + res);
                }
            }
        }

        // test: training using a attention mask block
        if (display_debug_info) cout << "\ntest: training attention models\n";
        {
            // Define the network
            int num_samples = (500 / mini_batch_size) * mini_batch_size;
            const int num_classes = 256;           
            const int num_epochs = 3000;

            // Shakespeare's text sample
            const string shakespeare_text = R"(HAMLET By William Shakespeare - Act Three, Scene One
To be or not to be—that is the question:
Whether ’tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And, by opposing, end them. To die, to sleep—
No more—and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to—’tis a consummation
Devoutly to be wished. To die, to sleep—
To sleep, perchance to dream. Ay, there’s the rub,
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There’s the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
Th’ oppressor’s wrong, the proud man’s contumely,
The pangs of despised love, the law’s delay,
The insolence of office, and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscovered country from whose bourn
No traveler returns, puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all,
And thus the native hue of resolution
Is sicklied o’er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry
And lose the name of action.—Soft you now,
The fair Ophelia.—Nymph, in thy orisons
Be all my sins remembered.)";

            using net_type_a = classification_head<num_classes,
                repeat<1, transformer_block,
                tag10<input<matrix<float>>>>>;
            net_type_a net_a;
            using net_type_b = classification_head<num_classes,
                repeat<2, transformer_block,
                positional_embeddings<num_classes, embedding_size,
                input<matrix<int, 0, 1>>>>>;
            net_type_b net_b;            

            // Generate synthetic training data
            dlib::rand rnd;
            std::vector<matrix<float>> samples;
            std::vector<unsigned long> labels;
            for (int i = 0; i < num_samples; ++i) {
                matrix<float> sample(sequence_size, embedding_size);
                for (int r = 0; r < sequence_size; ++r) {
                    for (int c = 0; c < embedding_size; ++c) sample(r, c) = rnd.get_random_float() * 2.0f - 1.0f;
                }
                samples.push_back(sample);
                labels.push_back(rnd.get_random_32bit_number() % num_classes);
            }

            // Split data into batches
            std::vector<std::vector<matrix<float>>> batches;
            std::vector<std::vector<unsigned long>> label_batches;
            for (int i = 0; i < num_samples; i += mini_batch_size) {
                std::vector<matrix<float>> batch_samples(samples.begin() + i, samples.begin() + i + mini_batch_size);
                std::vector<unsigned long> batch_labels(labels.begin() + i, labels.begin() + i + mini_batch_size);
                batches.push_back(batch_samples);
                label_batches.push_back(batch_labels);
            }
           
            // Train multihead attention model
            if (!skip_tests[10]) {
                dnn_trainer<net_type_a> trainer_b(net_a);
                trainer_b.set_learning_rate(learning_rate);
                trainer_b.set_min_learning_rate(min_learning_rate);
                trainer_b.set_mini_batch_size(mini_batch_size);
                trainer_b.be_verbose();
                trainer_b.set_iterations_without_progress_threshold(50);
                for (int epoch = 0; epoch < num_epochs && trainer_b.get_learning_rate() > trainer_b.get_min_learning_rate() && !g_interrupt_signal_received; ++epoch) {
                    for (size_t i = 0; i < batches.size(); ++i) trainer_b.train_one_step(batches[i], label_batches[i]);
                }
                trainer_b.get_net();
                net_a.clean();
                cout << "multihead attention model parameters: " << count_parameters(net_a) << endl;
                g_interrupt_signal_received = false;
                std::vector<unsigned long> predicted_labels_b = net_a(samples);
                int num_correct_b = 0;
                for (size_t i = 0; i < labels.size(); ++i) if (predicted_labels_b[i] == labels[i]) ++num_correct_b;
                double accuracy_b = static_cast<double>(num_correct_b) / labels.size();
                DLIB_TEST_MSG(accuracy_b > 0.8, "multihead attention model (accuracy: " + to_string(accuracy_b) + ")");
            }

            // "shakespeare" example
            if (!skip_tests[11]) {
                // Lambda function to convert a vector of integers to a string of unsigned chars
                auto to_unsigned_char_string = [](const matrix<int, 0, 1>& ints) -> string {
                    string result;
                    for (int v = 0; v < ints.nr(); ++v) result += static_cast<unsigned char>(ints(v, 0));
                    return result;
                };
                // Lambda function for tokenizing text
                auto tokenize_text = [](const string& text, int sequence_len) -> std::vector<matrix<int, 0, 1>> {
                    std::vector<matrix<int, 0, 1>> tokens;
                    if (text.size() > (sequence_len + 1)) {
                        for (int i = 0; i < (int)(text.size()) - (sequence_len + 1); ++i) {
                            matrix<int> sample(sequence_len, 1);
                            for (size_t j = 0; j < sequence_len; ++j) sample(j, 0) = static_cast<unsigned char>(text[i + j]);
                            tokens.push_back(sample);
                        }
                    }                                        
                    return tokens;
                };
                // Tokenize the Shakespeare text
                documents data(sequence_size, 0, true);
                data.load_text(shakespeare_text, false);
                std::vector<matrix<int, 0, 1>> samples_txt = tokenize_text(shakespeare_text, sequence_size);
                cout << "batch size: " << mini_batch_size << endl;
                cout << "samples used for the training: " << samples_txt.size() << endl;
                std::vector<unsigned long> labels_txt;
                for (size_t i = 0; i < samples_txt.size(); ++i) labels_txt.push_back(static_cast<unsigned long>(shakespeare_text[i + sequence_size])); // Next character as label              
                dnn_trainer<net_type_b, adam> trainer_c(net_b, adam(weight_decay, beta1, beta2), gpus);
                trainer_c.set_learning_rate(learning_rate);
                trainer_c.set_min_learning_rate(min_learning_rate);
                trainer_c.set_mini_batch_size(mini_batch_size);
                trainer_c.be_verbose();
                trainer_c.set_synchronization_file("llm_shakespeare_model_a.ckp", std::chrono::minutes(5));
                trainer_c.set_iterations_without_progress_threshold(500);
                std::vector<matrix<int, 0, 1>> samples;
                std::vector<unsigned long> labels;
                if (trainer_c.get_learning_rate() >= trainer_c.get_min_learning_rate()) {
                    while (trainer_c.get_learning_rate() >= trainer_c.get_min_learning_rate() && !g_interrupt_signal_received) {
                        if (data.generate_samples(mini_batch_size, samples, labels, false)) trainer_c.train_one_step(samples, labels);
                        else g_interrupt_signal_received = true;
                    }
                    trainer_c.get_net();
                    net_b.clean();
                    serialize("llm_shakespeare_model_a.dat") << net_b;
                    cout << "shakespeare model saved: llm_shakespeare_model_a.dat" << endl;
                    cout << "shakespeare model parameters: " << count_parameters(net_b) << endl;
                    g_interrupt_signal_received = false;

                    // Test the network with the same data to ensure it has learned something
                    std::vector<unsigned long> predicted_labels_c = net_b(samples_txt);
                    int num_correct_c = 0;
                    for (size_t i = 0; i < labels_txt.size(); ++i) if (predicted_labels_c[i] == labels_txt[i]) ++num_correct_c;
                    double accuracy_c = static_cast<double>(num_correct_c) / labels_txt.size();
                    DLIB_TEST_MSG(accuracy_c > 0.8, "shakespeare model (accuracy: " + to_string(accuracy_c) + ")");
                }
                // Predict the next sequence of characters
                string input_sequence = "To be or not to be—that is the ques";
                std::vector<matrix<int, 0, 1>> input_tokens = tokenize_text(input_sequence, sequence_size);
                string start_seq = to_unsigned_char_string(input_tokens.back());
                size_t pos = input_sequence.find(start_seq);
                if (pos != std::string::npos) input_sequence = input_sequence.substr(0, pos + start_seq.length());
                cout << "input sequence for text generation: <" << start_seq << ">" << endl;
                matrix<int> next_input(sequence_size, 1);
                for (int i = 0; i < 400; ++i) {
                    unsigned long next_char = net_b(input_tokens.back());
                    input_sequence += static_cast<unsigned char>(next_char);
                    for (int j = 0; j < (sequence_size - 1); ++j) next_input(j, 0) = input_tokens.back()(j + 1, 0);
                    next_input(sequence_size - 1, 0) = static_cast<int>(next_char);
                    input_tokens.clear();
                    input_tokens.push_back(next_input);
                }
                cout << "generated text:\n\n" << input_sequence << " (...)\n\n";

                // Loading now the complete Shakespeare file
                string shakespeare_file = "shakespeare.txt";
                if (fs::exists(shakespeare_file)) {
                    documents shakespeare_data(sequence_size, 0, true);
                    shakespeare_data.load_documents(shakespeare_file, false);
                    cout << "loaded " << shakespeare_data.get_total_tokens() << " tokens from " << shakespeare_file << endl;

                    // Reload previous model
                    if (!fs::exists("llm_shakespeare_model_b.ckp")) {
                        if (!fs::exists("llm_shakespeare_model_b.dat") && fs::exists("llm_shakespeare_model_a.dat")) {
                            deserialize("llm_shakespeare_model_a.dat") >> net_b;
                            cout << "shakespeare model loaded (source template): llm_shakespeare_model_a.dat" << endl;
                        } else if (fs::exists("llm_shakespeare_model_b.dat")) {
                            deserialize("llm_shakespeare_model_b.dat") >> net_b;
                            cout << "shakespeare model loaded: llm_shakespeare_model_b.dat" << endl;
                        } else {
                            cout << "no previous model found, starting from scratch" << endl;
                        }
                    } else {
                        cout << "restarting from from the last checkpoint" << endl;
                    }
                    dnn_trainer<net_type_b, adam> trainer_d(net_b, adam(weight_decay, beta1, beta2), gpus);
                    trainer_d.set_learning_rate(learning_rate);
                    trainer_d.set_min_learning_rate(min_learning_rate);
                    trainer_d.set_mini_batch_size(mini_batch_size);
                    trainer_d.be_verbose();                    
                    trainer_d.set_synchronization_file("llm_shakespeare_model_b.ckp", std::chrono::minutes(5));
                    trainer_d.set_iterations_without_progress_threshold(iterations_without_progress_threshold);

                    // New training loop
                    while (trainer_d.get_learning_rate() >= trainer_d.get_min_learning_rate() && !g_interrupt_signal_received) {
                        if (shakespeare_data.generate_samples(mini_batch_size, samples, labels)) trainer_d.train_one_step(samples, labels);                        
                        else g_interrupt_signal_received = true;
                    }
                    trainer_d.get_net();
                    net_b.clean();
                    serialize("llm_shakespeare_model_b.dat") << net_b;
                    cout << "advanced shakespeare model saved: llm_shakespeare_model_b.dat" << endl;

                    // Attempting to generate a new sonnet
                    string sonnet_start = "Shall I compare thee to a winter's night?";
                    std::vector<matrix<int, 0, 1>> input_tokens = tokenize_text(sonnet_start+"\n", sequence_size);
                    if (!input_tokens.empty()) {
                        string generated_sonnet = sonnet_start;
                        matrix<int> next_input(sequence_size, 1);

                        cout << "generated sonnet starting by \"" << sonnet_start << "\":\n\n" << sonnet_start << "\n";
                        for (int i = 0; i < 700 && !input_tokens.empty(); ++i) {
                            unsigned long next_char = net_b(input_tokens.back());
                            unsigned char c = static_cast<unsigned char>(next_char);
                            generated_sonnet += c;
                            cout << c;
                            if (c == '\n') {
                                generated_sonnet += '\n';  // Double newline for readability
                                cout << "\n";
                            }

                            for (int j = 0; j < (sequence_size - 1); ++j) next_input(j, 0) = input_tokens.back()(j + 1, 0);
                            next_input(sequence_size - 1, 0) = static_cast<int>(next_char);

                            input_tokens.clear();
                            input_tokens.push_back(next_input);

                            // Stop after generating what looks like a complete sonnet
                            if (generated_sonnet.find("END") != string::npos || generated_sonnet.find("\n\n\n\n") != string::npos) break;
                        }
                        cout << endl;

                        // Basic relevance test
                        std::vector<string> keywords = {
                            "thou", "thy", "thee", "love", "beauty", "time", "death", "life",
                            "king", "lord", "heart", "good", "night", "day", "man", "great",
                            "eyes", "sweet", "fair", "world", "hand", "heaven", "father", "blood",
                            "mind", "know", "make", "god", "son", "well", "long", "come",
                            "hand", "art", "young", "dear", "true", "friend", "honour", "bear",
                            "give", "lady", "sir", "queen", "speak", "face", "court", "live",
                            "say", "soul", "leave", "heart", "grace", "power", "nature", "truth",
                            "fear", "noble", "crown", "sword", "head", "hear", "stand", "tongue",
                            "never", "light", "name", "peace", "hell", "spirit", "body", "master",
                            "word", "poor", "prince", "fortune", "hope", "virtue", "law", "tale"
                        };                        int keyword_count = 0;
                        for (const auto& keyword : keywords) {
                            if (generated_sonnet.find(keyword) != string::npos) keyword_count++;
                        }
                        double relevance_score = static_cast<double>(keyword_count) / keywords.size();
                        DLIB_TEST_MSG(relevance_score > 0.3, "shakespeare model relevance (score: " + to_string(relevance_score) + ")");
                    } else {
                        cout << "error: unable to tokenize sonnet start" << endl;
                    }
                } else {
                    cout << "error: shakespeare.txt file not found in the current directory" << endl;
                }
            }
        }
    } else if (text_generation) {
        if (fs::exists(vocabulary_prefix + ".model")) status = sp.Load(vocabulary_prefix + ".model");
        else {
            cerr << "vocabulary file not found! (<" << (vocabulary_prefix + ".model|.vocab") << ">)" << endl;
            return 1;
        }
        llm_net net;
        softmax<multiply<llm_net::subnet_type>> generator(multiply_(1.0 / temperature));
        if (fs::exists(language_model)) deserialize(language_model) >> net;
        else {
            cerr << "language model not found! (<" << language_model << ">)" << endl;
            return 1;
        }
        generator.subnet().subnet() = net.subnet();
        cout << "number of model parameters: " << count_parameters(generator) << endl << endl;
        context_window prompt(sequence_size);
        string input = "The salesperson", output = "";
        cout << "Input prompt: " << input << " (...)" << endl;
        cout << "Generated text: " << input << " ";

        std::vector<int> prompt_ids, endings = { eos_id, pad_id }, response_ids;
        sp.Encode(dlib::trim(input), &prompt_ids);
        prompt.add_input(prompt_ids);
        matrix<int, 0, 1> padded_window;
        for (int i = 0; i < 100; ++i) {
            if (prompt.get_padded_window(padded_window)) {
                matrix<float, vocab_size, 1> logits = mat(generator(padded_window));
                int predicted_label = index_of_max(logits);
                prompt.add_output(predicted_label);
                response_ids.push_back(predicted_label);
            }
        }
        sp.Decode(response_ids, &output);
        cout << output << endl;
        return 0;
    } else if (voc_training) {
        /*{
            string initial_raw_data = corpus_dir + "/internal_raw_data.txt";
            write_raw_data(initial_raw_data);
            concatenate_files(corpus_dir);
            return 1;
        }*/
        std::vector<int> vocab_sizes = { 3000, 8000, 20000, 40000 };
        string corpus_files;
        for (const auto& entry : fs::recursive_directory_iterator(corpus_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                corpus_files += "\"" + entry.path().string() + "\",";
            }
        }
        corpus_files.pop_back();

        for (const auto& vocab_size : vocab_sizes) {
            string size_suffix;
            if (vocab_size == 3000) size_suffix = "3k";
            else if (vocab_size == 8000) size_suffix = "8k";
            else if (vocab_size == 20000) size_suffix = "20k";
            else if (vocab_size == 40000) size_suffix = "40k";
            string current_vocabulary_prefix = "ernie.en-fr.ung." + size_suffix;

            string train_args = "--input=" + corpus_files +
                " --model_prefix=" + current_vocabulary_prefix +
                " --bos_id=" + to_string(bos_id) + " --eos_id=" + to_string(eos_id) +
                " --unk_id=" + to_string(unk_id) + " --pad_id=" + to_string(pad_id) +
                " --model_type=unigram" +
                " --character_coverage=1.0" +
                " --max_sentence_length=16768" +
                " --split_by_unicode_script=false" +
                " --input_sentence_size=3500000" +
                " --shuffle_input_sentence=true" +
                " --train_extremely_large_corpus=true" +
                " --vocab_size=" + to_string(vocab_size);

            auto status = sentencepiece::SentencePieceTrainer::Train(train_args);
            if (!status.ok()) {
                cerr << "error training tokenizer " << current_vocabulary_prefix << ": " << status.message() << endl;
                return 1;
            } else {
                cout << "successfully trained tokenizer " << current_vocabulary_prefix << endl;
            }
        }
    } else if (model_training) {
        if (fs::exists(vocabulary_prefix + ".model")) status = sp.Load(vocabulary_prefix + ".model");
        else {
            cerr << "vocabulary file not found! (<" << (vocabulary_prefix + ".model|.vocab") << ">)" << endl;
            return 1;
        }
        
        const string model_sync_filename = fs::current_path().string() + "/ernie_checkpoint.dat";        
        llm_net net;
        adam solver(weight_decay, beta1, beta2);
        dnn_trainer<llm_net, adam> my_trainer(net, solver, gpus);
        my_trainer.set_learning_rate(learning_rate);
        my_trainer.set_min_learning_rate(min_learning_rate);
        my_trainer.set_iterations_without_progress_threshold(iterations_without_progress_threshold);
        my_trainer.set_mini_batch_size(mini_batch_size);
        if (use_sync_file) my_trainer.set_synchronization_file(model_sync_filename, std::chrono::minutes(5));
        my_trainer.be_verbose();
        if (!fs::exists(model_sync_filename) && fs::exists(language_model)) deserialize(language_model) >> net;
        std::ostringstream oss;
        oss << net << endl << my_trainer;
        cout << oss.str() << endl;
        
        documents data;
        std::vector<matrix<int, 0, 1>> samples;
        std::vector<unsigned long> labels;
        cout << "preprocessing entries... " << endl;
        if (corpus_dir.empty()) {
            string initial_raw_data = fs::current_path().string() + "/ernie_raw_data.txt";
            write_raw_data(initial_raw_data);
            data.load_documents(initial_raw_data);
            fs::remove(initial_raw_data);
        } else data.load_documents(corpus_dir, false);
        cout << data.get_total_tokens() << " tokens for the training" << endl;

        // Training loop
        while (!g_interrupt_signal_received && my_trainer.get_learning_rate() >= my_trainer.get_min_learning_rate()) {
            if (data.generate_samples(mini_batch_size, samples, labels)) my_trainer.train_one_step(samples, labels);
            else g_interrupt_signal_received = true;            
        }
        cout << "stopping the training process" << endl;
        my_trainer.get_net();
        net.clean();
        serialize(language_model) << net;
        //dlib::net_to_xml(net, "for_debug.xml");
        cout << endl << "language model <" << language_model << "> saved" << endl;
        if (use_sync_file && !g_interrupt_signal_received) {
            fs::remove(model_sync_filename);
            fs::remove((string(model_sync_filename) + string("_")).c_str());
        }
    } else if (model_prompting) {
        // TEST: context_window class
        /*
        {
            context_window prompt(20);
            std::vector<int> input = { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };
            std::cout << "input vector: ";
            std::copy(input.begin(), input.end(), std::ostream_iterator<int>(std::cout, " "));
            std::cout << std::endl;
            prompt.add_input(input);
            matrix<int, 0, 1> padded_window;
            for (size_t i = 0; i < 60; i++) {
                if (prompt.get_padded_window(padded_window)) {
                    std::cout << "padded window (i=" << i << "): " << padded_window;
                    std::cout << std::endl;
                }
                if ((i % 4) == 0) prompt.add_output(0, true);
                if (i == 3) prompt.add_input(input, true);
                if (i == 15) prompt.add_input(input);
                if (i == 30) prompt.add_input(input, true);
            }
            return 1;
        }
        */
        if (fs::exists(vocabulary_prefix + ".model")) status = sp.Load(vocabulary_prefix + ".model");
        else {
            cerr << "vocabulary file not found! (<" << (vocabulary_prefix + ".model|.vocab") << ">)" << endl;
            return 1;
        }
        llm_net net;
        softmax<multiply<llm_net::subnet_type>> generator(multiply_(1.0 / temperature));
        if (fs::exists(language_model)) deserialize(language_model) >> net;
        else {
            cerr << "language model not found! (<" << language_model << ">)" << endl;
            return 1;
        }
        generator.subnet().subnet() = net.subnet();
        cout << "number of model parameters: " << count_parameters(generator) << endl;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(sequence_size/2, sequence_size*6);
        context_window prompt(sequence_size);
        std::vector<int> prompt_ids, endings = { eos_id, pad_id }, response_ids;

        cout << ">>> press [CTRL+C] to stop the dialog with ERNIE <<<" << endl << endl;
        string input, output;
        cout << "[YOU] ";
        std::getline(std::cin, input);
        do {            
            if (!g_interrupt_signal_received && !input.empty()) {
                sp.Encode(dlib::trim(input), &prompt_ids);
                prompt.reset();
                prompt.add_input(prompt_ids);
                size_t total_steps = std::min(static_cast<int>(sequence_size), dis(gen));
                // Generate response
                response_ids.clear();
                cout << "[ERNIE] ";
                matrix<int, 0, 1> padded_window;
                int cur_top_k = (top_k >= 1) ? top_k : 1, predicted_label;
                for (int i = 0; i < total_steps; ++i) {                    
                    if (prompt.get_padded_window(padded_window)) {
                        //cout << padded_window << endl;
                        matrix<float, vocab_size, 1> logits = mat(generator(padded_window));
                        //cout << "logits.nr()=" << logits.nr() << " - logits.nc()=" << logits.nc() << endl;
                        if (cur_top_k <= 1) {
                            predicted_label = index_of_max(logits);
                            //cout << "logits=" << logits << endl;
                        } else {                                                        
                            std::vector<float> top_k_probs(cur_top_k);
                            std::vector<int> top_k_indices(cur_top_k);
                            for (int k = 0; k < cur_top_k; ++k) {
                                predicted_label = index_of_max(logits);                                
                                top_k_indices[k] = predicted_label;
                                top_k_probs[k] = logits(predicted_label);
                                logits(predicted_label) = 0.0f;
                            }
                            std::discrete_distribution<> top_k_distribution(top_k_probs.begin(), top_k_probs.end());
                            predicted_label = top_k_indices[top_k_distribution(gen)];
                            cur_top_k--;
                        }
                        if (response_ids.size() > (total_steps / 3) && std::find(std::begin(endings), std::end(endings), predicted_label) != std::end(endings)) break;
                        prompt.add_output(predicted_label);
                        response_ids.push_back(predicted_label);                                            
                    } else break;
                }
                output.clear();
                sp.Decode(response_ids, &output);
                if (!output.empty()) cout << output;
                cout << endl << endl << "[YOU] ";                
                input.clear();
                std::getline(std::cin, input);
            }            
        } while (!g_interrupt_signal_received);
        cout << "[ERNIE] Au revoir !" << endl;
    }
}