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

#include "llm_defs.h"
#include "advanced_tokenizer.hpp"
#include "data_fr.h"

#include <sentencepiece_trainer.h>
#include <sentencepiece_processor.h>

namespace fs = std::filesystem;
namespace po = boost::program_options;

const int bos_id = 0, eos_id = 1, unk_id = 2, pad_id = 3;

// Other global parameters
string vocabulary_prefix = "ernie.eu.ung.12k", language_model = "ernie_vslm_v1.dat";
std::unique_ptr<advanced_tokenizer> tokenizer_;

void configure_console() {
    SetConsoleOutputCP(CP_UTF8);
    int res = _setmode(_fileno(stdout), _O_TEXT);
    if (res == -1) cerr << "Cannot set mode" << endl;
    cout.imbue(std::locale("en_US.UTF-8"));    
}

namespace utils {
    string replace_html_entities(const string& input) {
        static const std::unordered_map<string, string> htmlEntities = {
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

        string output;
        output.reserve(input.size());

        size_t lastPos = 0;
        size_t findPos = 0;

        while ((findPos = input.find('&', lastPos)) != string::npos) {
            output.append(input, lastPos, findPos - lastPos);
            auto endPos = input.find(';', findPos);
            if (endPos != string::npos) {
                string entity = input.substr(findPos, endPos - findPos + 1);
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
        output.append(input, lastPos, string::npos);
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

    bool is_utf8(const string& s) {
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
            cerr << "Error opening output file: " << output_file << endl;
            return;
        }

        for (const auto& entry : fs::recursive_directory_iterator(directory_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                std::ifstream input(entry.path(), std::ios::binary);
                if (!input) {
                    cerr << "Error opening input file: " << entry.path() << endl;
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

const size_t std_global_context_size = (5 * llm::sequence_size);
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
    documents(size_t seq_size = llm::sequence_size, int pad_value = pad_id,
        bool use_letter_tokenization = false, size_t token_limit = 25000) :
        sequence_size_(seq_size), pad_value_(pad_value), use_letter_tokenization_(use_letter_tokenization), token_limit_(token_limit) {
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
    size_t get_total_samples(void) { return total_tokens_ > sequence_size_ ? (total_tokens_ - sequence_size_) : 0; }
    size_t get_total_presamples(void) { return pre_samples_.size(); }
    void clear_all(void) {        
        source_tokens_.clear();
        pre_samples_.clear();
        pre_labels_.clear();
        total_tokens_ = 0;
        samples_idx_ = 0;
    }

    void load_text(const string& text, bool split_sentences) {
        if (!is_initialized_) return;

        std::vector<std::vector<int>> new_tokens;
        size_t current_total_tokens = total_tokens_;

        auto process_tokens = [&](const std::vector<int>& tokens) {
            if (tokens.empty()) return;

            if (current_total_tokens + tokens.size() <= token_limit_) {
                new_tokens.push_back(tokens);
                current_total_tokens += tokens.size();
            }
            else if (current_total_tokens < token_limit_) {
                size_t remaining_space = token_limit_ - current_total_tokens;
                std::vector<int> truncated_tokens(tokens.begin(), tokens.begin() + remaining_space);
                new_tokens.push_back(truncated_tokens);
                current_total_tokens = token_limit_;
            }
        };

        if (split_sentences) {
            std::vector<string> sentences = split_into_sentences(text);
            for (const auto& sentence : sentences) {
                if (current_total_tokens >= token_limit_) break;
                std::vector<int> tokens = preprocess_sentence(sentence);
                process_tokens(tokens);
            }
        }
        else {
            std::vector<int> tokens = preprocess_sentence(text);
            process_tokens(tokens);
        }

        // Update the source_tokens_ and total_tokens_
        source_tokens_.insert(source_tokens_.end(), new_tokens.begin(), new_tokens.end());
        total_tokens_ = current_total_tokens;

        if (pre_samples_.size() > 0) {
            pre_samples_.clear();
            pre_labels_.clear();
            samples_idx_ = 0;
        }
    }

    void load_documents(const string& path, bool split_sentences = true) {
        if (!is_initialized_) return;
        fs::path fs_path = fs::path(path);
        try {
            if (fs::is_regular_file(fs_path) && fs_path.extension() == ".txt") {
                cout << "loading file: " << fs_path.string() << endl;
                std::ifstream file(fs_path, std::ios::binary);
                string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
                if (!is_utf8(content)) cout << "warning - file <" << fs_path.string() << "> seems not to be UTF-8 encoded" << endl;
                load_text(content, split_sentences);
            } else if (fs::is_directory(fs_path)) {
                for (const auto& entry : fs::recursive_directory_iterator(fs_path)) {
                    if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                        cout << "loading file: " << entry.path().string() << endl;
                        std::ifstream file(entry.path(), std::ios::binary);
                        string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
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
                for (const auto& sentence : source_tokens_) {                    
                    if (sentence.size() > (sequence_size_ + 1)) {
                        size_t i, j;
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
    size_t total_tokens_, token_limit_;
    int pad_value_;
    bool use_letter_tokenization_;
    bool is_initialized_;

    std::vector<string> split_into_sentences(const string& text) {
        std::vector<string> sentences = dlib::split(text, "\r\n");
        return sentences;
    }

    std::vector<int> preprocess_sentence(const string& sentence, bool add_eos_id = false) {
        string cleaned_sentence = dlib::trim(replace_html_entities(std::regex_replace(sentence, std::regex("(.)\\1{4,}"), "$1$1$1$1")));
        std::vector<int> tokens;
        if (!use_letter_tokenization_) {
            sp_.Encode(cleaned_sentence, &tokens);
        } else {
            for (size_t i = 0; i < sentence.size(); ++i) tokens.push_back(static_cast<unsigned char>(sentence[i]));
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
    tt::tensor_rand rnd(std::rand());
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
    tt::tensor_rand rnd(std::rand());
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
    tt::tensor_rand rnd(std::rand());
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
    DLIB_TEST_MSG(max(abs(mat(net_output) - expected_output)) < 1e-5, "positional_encodings layer");
}

void test_embeddings()
{
    const size_t num_sequences = 100, sequence_length = 7, num_classes = 3, num_tokens = 50, embedding_length = 5;
    using net_type = loss_multiclass_log<fc<num_classes,
        relu<fc<32,relu<fc<64,
        embeddings<num_tokens, embedding_length,
        input<matrix<unsigned long, 0, 1>>>>>>>>>;
    net_type net;
    dnn_trainer<net_type> trainer(net, sgd(0, 0.9));
    trainer.set_learning_rate(1e-1);
    trainer.set_min_learning_rate(1e-4);
    trainer.set_mini_batch_size(16);
    trainer.set_max_num_epochs(500);

    dlib::rand rnd(std::rand());
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
        if (predicted_labels[i] == labels[i]) ++num_correct;
    
    double acc = static_cast<double>(num_correct) / labels.size();
    DLIB_TEST_MSG(acc > 0.9, "embeddings accuracy: " + to_string(acc));
}

void test_rms_normalize()
{
    resizable_tensor x(2, 3, 4, 5);
    resizable_tensor y_cpu(x);
    tt::tensor_rand rnd(std::rand());
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
    dlib::rand rnd(std::rand());
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

void test_multm_prev()
{
    //print_spinner();
    using net_type = tag1<multm_prev6<skip5<tag6<transpose<tag5<input<matrix<float>>>>>>>>;
    net_type net;

    // Input tensor
    dlib::rand rnd(std::rand());
    const int nr = 3, nc = 4;
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

    resizable_tensor expected_output(n_samples, k, nr, nr);
    matrix<float> input_mat(nr, nc);
    matrix<float> output_mat(nr, nr);

    for (long s = 0; s < n_samples; ++s) {
        for (long r = 0; r < nr; ++r) {
            for (long c = 0; c < nc; ++c) {
                input_mat(r, c) = input_tensor.host()[tensor_index(input_tensor, s, 0, r, c)];
            }
        }
        output_mat = input_mat * trans(input_mat);

        for (long r = 0; r < nr; ++r) {
            for (long c = 0; c < nr; ++c) {
                expected_output.host()[tensor_index(expected_output, s, 0, r, c)] = output_mat(r, c);
            }
        }
    }

    auto& net_output = layer<tag1>(net).get_output();
    DLIB_TEST_MSG(max(abs(mat(net_output) - mat(expected_output))) < 1e-5, "multm_prev layer");
}

void test_softmaxm()
{
    //print_spinner();
    using net_type = tag1<softmaxm<tag2<input<matrix<float>>>>>;
    net_type net;

    // Initialization
    dlib::rand rnd(std::rand());
    const long nr = 2, nc = 3;
    const int n_samples = 3, k = 1;
    std::vector<matrix<float>> x(n_samples);
    matrix<float> xtmp(nr, nc);
    for (int ii = 0; ii < n_samples; ++ii) {
        for (int jj = 0; jj < nr; ++jj)
            for (int kk = 0; kk < nc; ++kk) {
                float r = rnd.get_random_gaussian();
                if (r > 1 || r < -1) r = -std::numeric_limits<float>::infinity();
                xtmp(jj, kk) = r;
            }
        x[ii] = xtmp;
    }

    // Convert input matrix to tensor
    resizable_tensor input_tensor;
    net.to_tensor(&x[0], &x[0] + n_samples, input_tensor);
    net.forward(input_tensor);

    // Expected output tensor
    resizable_tensor expected_output;
    expected_output.copy_size(input_tensor);
    for (int ii = 0; ii < n_samples; ++ii) {
        for (int jj = 0; jj < nr; ++jj) {
            matrix<float> m(1, nc);
            bool all_neg_inf = true;
            for (int kk = 0; kk < nc; ++kk) {
                m(0, kk) = input_tensor.host()[tensor_index(input_tensor, ii, 0, jj, kk)];
                if (m(0, kk) > -std::numeric_limits<float>::infinity()) all_neg_inf = false;
            }

            matrix<float> r(1, nc);
            if (all_neg_inf)
                for (int kk = 0; kk < nc; ++kk) r(0, kk) = 0.0f;
            else {
                // Stabilize the computation by subtracting the max value
                float max_val = max(m);
                matrix<float> exp_m = exp(m - max_val);
                float sum_exp = sum(exp_m) + std::numeric_limits<float>::epsilon();
                r = exp_m / sum_exp;
            }
            for (int kk = 0; kk < nc; ++kk)
                expected_output.host()[tensor_index(expected_output, ii, 0, jj, kk)] = r(0, kk);
        }
    }

    // Compare output tensor with expected output
    auto& net_output = layer<tag1>(net).get_output();
    DLIB_TEST_MSG(max(abs(mat(net_output) - mat(expected_output))) < 1e-5, "softmaxm layer");

    // Compare CPU and CUDA utility functions
    resizable_tensor output_tensor, cpu_grad, gradient_input;
    output_tensor.copy_size(input_tensor);
    cpu_grad.copy_size(input_tensor);
    cpu_grad = 0;
    gradient_input.copy_size(input_tensor);    
    randomize_parameters(gradient_input, nr + nc, rnd);
    cpu::softmax(output_tensor, input_tensor, 1);    
    cpu::softmax_gradient(cpu_grad, output_tensor, gradient_input, 1);
    DLIB_TEST_MSG(max(abs(mat(output_tensor) - mat(expected_output))) < 1e-5, "softmax (cpu)");
#ifdef DLIB_USE_CUDA
    resizable_tensor cuda_grad;
    cuda_grad.copy_size(input_tensor);
    cuda_grad = 0;
    cuda::softmax(output_tensor, input_tensor, 1);
    cpu::softmax_gradient(cuda_grad, output_tensor, gradient_input, 1);
    DLIB_TEST_MSG(max(abs(mat(output_tensor) - mat(expected_output))) < 1e-5, "softmax (cuda)");
    DLIB_TEST_MSG(max(abs(mat(cuda_grad) - mat(cpu_grad))) < 1e-5, "softmax_gradient cpu-cuda");
#endif
}

void test_linear()
{
    // Define the network
    using net_type = tag2<linear_no_bias<6, tag1<input<matrix<float>>>>>;
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
    matrix<float> expected_output = input_matrix * w;

    // Compare output tensor with expected output
    auto& net_output = layer<tag2>(net).get_output();

    // Display results
    DLIB_TEST_MSG(max(abs(mat(net_output) - expected_output)) < 1e-5, "linear layer");
}

int main(int argc, char* argv[]) {
    string corpus_dir;
    bool do_benchmark = false, text_generation = false;
    bool voc_training = false, model_training = false, model_prompting = false, use_sync_file = false;
    double learning_rate = 1e-3, min_learning_rate = 1e-6, weight_decay = 0.05, beta1 = 0.9, beta2 = 0.999, temperature = 0.9;
    long mini_batch_size = 64, iterations_without_progress_threshold = 50000, top_k = 3;
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
            ("corpus-directory,d", po::value<string>(&corpus_dir), "Directory containing text files to process")
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
            false,      // 0: strings & tokenization
            false,      // 1: transpose layer
            false,      // 2: tril layer
            false,      // 3: positional_encodings layer
            false,      // 4: embeddings layer
            false,      // 5: multm_prev layer
            false,      // 6: softmax layer
            false,      // 7: attention mechanism
            false,      // 8: linear layer         
            true,       // 9: hsplit/hstack layers
            false,      // 10: rms_norm layer
            true,      // 11: multihead attention model
            false       // 12: "shakespeare" example
        };

        // test: tokenization
        if (!skip_tests[0]) {
            if (display_debug_info) cout << "test: strings & tokenization\n";
            string sentence = "  &nbsp;&lt;p&gt;Hellooooooo     frieeeends !!!!!! This is sooooooo coooool &amp; awesoooooome !&lt;/p&gt;  ";
            string cleaned_sentence = dlib::trim(replace_html_entities(std::regex_replace(sentence, std::regex("(.)\\1{4,}"), "$1$1$1$1")));
            cout << "string normalisation: [" << sentence << "] => [" << cleaned_sentence << "]" << endl;

            if (fs::exists(vocabulary_prefix + ".model")) status = sp.Load(vocabulary_prefix + ".model");
            else {
                cerr << "vocabulary file not found! (<" << (vocabulary_prefix + ".model|.vocab") << ">)" << endl;
            }
            std::vector<string> test_sentences = {
                "This is a test sentence in English.",
                "Ceci est une phrase de test en français.</s>",
                "Dies ist ein Testsatz auf Deutsch.",
                "<s>Questa è una frase di prova in italiano.</s>",
                "Esta es una frase de <unk> en español."
            };
            for (const auto& sentence : test_sentences) {
                std::vector<string> tokens;
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

        // test: multm_prev layer
        if (!skip_tests[5]) {
            if (display_debug_info) cout << "test: multm_prev layer\n";
            test_multm_prev();            
        }

        // test: softmax layer
        if (!skip_tests[6]) {
            if (display_debug_info) cout << "\ntest: softmax layer\n";
            test_softmaxm();
            {
                softmax2_<softmax_mode::CHANNEL_WISE> l;
                auto res = test_layer(l);
                DLIB_TEST_MSG(res, " softmaxm test_0 layer\n" + res);
            }
            {
                softmax2_<softmax_mode::PLANE_WISE> l;
                auto res = test_layer(l);
                DLIB_TEST_MSG(res, " softmaxm test_1 layer\n" + res);
            }
        }

        // test: attention mechanism
        if (!skip_tests[7]) {
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
                            for (long c = 0; c < m.nc(); ++c) result(r, c) = 0.0f;
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
                    cout << "Q:\n" << Q << endl;
                    cout << "K:\n" << K << endl;
                    cout << "V:\n" << V << endl;
                    cout << "scores:\n" << scores << endl;
                    cout << "attention weights (softmax):\n" << attention_weights << endl;
                    cout << "Z:\n" << Z << endl;
                }

                // Model definition
                using net_type = tag10<multm_prev1<tag7<softmaxm<llm::scale_weights<3, tag6<multm_prev4<
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
        if (!skip_tests[8]) {
            if (display_debug_info) cout << "\ntest: linear layer\n";
            test_linear();
            {
                linear_<4, LINEAR_NO_BIAS> l;
                auto res = test_layer(l);
                DLIB_TEST_MSG(res, " linear test_0 layer\n" + res);
            }
            {
                linear_<3, LINEAR_HAS_BIAS> l;
                auto res = test_layer(l);
                DLIB_TEST_MSG(res, " linear test_1 layer\n" + res);
            }
        }

        // test: hsplit/hstack layers
        if (!skip_tests[9]) {
            if (display_debug_info) cout << "\ntest: hsplit/hstack layers\n";                            
            test_hsplit_hstack();
            {
                hstack_ l;
                auto res = test_layer(l);
                DLIB_TEST_MSG(res, " hsplit/hstack layers\n" + res);
            }
        }

        // test: rms_norm layer
        if (!skip_tests[10]) {
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
            int num_samples = (2500 / mini_batch_size) * mini_batch_size;
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

            // Custom values for the local assessment
            using net_type_a = llm::classification_head<num_classes,                
                llm::v1_1_4::transformer<llm::sequence_size, llm::embedding_size, llm::number_of_heads,
                input<matrix<float>>>>;            
            using net_type_b = llm::classification_head<num_classes,
                llm::v1_1_4::transformer<llm::sequence_size, llm::embedding_size, llm::number_of_heads,
                llm::positional_embeddings<num_classes, llm::embedding_size,
                input<matrix<int, 0, 1>>>>>;            

            // Generate synthetic training data
            dlib::rand rnd(std::rand());
            std::vector<matrix<float>> samples;
            std::vector<unsigned long> labels;
            for (int i = 0; i < num_samples; ++i) {
                matrix<float> sample(llm::sequence_size, llm::embedding_size);
                for (int r = 0; r < llm::sequence_size; ++r) {
                    for (int c = 0; c < llm::embedding_size; ++c) sample(r, c) = rnd.get_random_float() * 2.0f - 1.0f;
                }
                samples.push_back(sample);
                labels.push_back(rnd.get_random_32bit_number() % num_classes);
            }
            auto count_unique_classes = [&labels]() {
                std::unordered_set<unsigned long> unique_classes(labels.begin(), labels.end());
                return unique_classes.size();
            };

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
            if (!skip_tests[11]) {
                cout << "number of samples: " << samples.size() << endl;
                cout << "number of classes: " << count_unique_classes() << std::endl;
                cout << "number of sample batches: " << batches.size() << endl;
                cout << "number of label batches: " << label_batches.size() << endl;

                net_type_a net_a;
                dnn_trainer<net_type_a> trainer_b(net_a, sgd(weight_decay, beta1), gpus);
                trainer_b.set_learning_rate(learning_rate);
                trainer_b.set_min_learning_rate(min_learning_rate);
                trainer_b.set_mini_batch_size(mini_batch_size);
                trainer_b.be_verbose();
                trainer_b.set_iterations_without_progress_threshold(150);
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
            if (!skip_tests[12]) {
                // Lambda function to convert a vector of integers to a string of unsigned chars
                auto to_unsigned_char_string = [](const matrix<int, 0, 1>& ints) -> string {
                    string result;
                    for (int v = 0; v < ints.nr(); ++v) result += static_cast<unsigned char>(ints(v, 0));
                    return result;
                };

                // Lambda function for tokenizing text
                auto tokenize_text = [](const string& text, int sequence_len, char padding_char = pad_id) -> std::vector<matrix<int, 0, 1>> {
                    std::vector<matrix<int, 0, 1>> tokens;
                    if (text.empty()) return tokens;

                    if (text.size() <= sequence_len) {
                        matrix<int> sample(sequence_len, 1);
                        sample = static_cast<int>(padding_char);
                        for (size_t i = 0; i < text.size(); ++i) sample(i, 0) = static_cast<unsigned char>(text[i]);
                        tokens.push_back(sample);
                    }
                    else {
                        for (size_t i = 0; i < static_cast<int>(text.size()) - (sequence_len + 1); ++i) {
                            matrix<int> sample(sequence_len, 1);
                            for (size_t j = 0; j < sequence_len; ++j) sample(j, 0) = static_cast<unsigned char>(text[i + j]);
                            tokens.push_back(sample);
                        }
                    }
                    return tokens;
                };

                // Tokenize the Shakespeare text
                std::vector<matrix<int, 0, 1>> samples;
                std::vector<unsigned long> labels;
                documents data(llm::sequence_size, pad_id, true);
                data.load_text(shakespeare_text, false);
                std::vector<matrix<int, 0, 1>> samples_txt = tokenize_text(shakespeare_text, llm::sequence_size);
                cout << "batch size: " << mini_batch_size << endl;
                cout << "espected number of samples: " << samples_txt.size() << endl;
                data.generate_samples(mini_batch_size, samples, labels, false);
                cout << "number of generated samples: " << data.get_total_presamples() << endl;
                cout << "number of sample batches: " << (data.get_total_presamples() / mini_batch_size) << endl;
                std::vector<unsigned long> labels_txt;
                for (size_t i = 0; i < samples_txt.size(); ++i) {
                    if (i + llm::sequence_size < shakespeare_text.length()) {                        
                        labels_txt.push_back(static_cast<unsigned long>(shakespeare_text[i + llm::sequence_size])); // Next character as label
                    }
                }
                
                net_type_b net_b;
                adam solver(weight_decay, beta1, beta2);
                dnn_trainer<net_type_b, adam> trainer_c(net_b, solver, gpus);
                trainer_c.set_learning_rate(learning_rate);
                trainer_c.set_min_learning_rate(min_learning_rate);
                trainer_c.set_mini_batch_size(mini_batch_size);
                trainer_c.be_verbose();
                trainer_c.set_synchronization_file("llm_shakespeare_model_a.ckp", std::chrono::minutes(5));
                trainer_c.set_iterations_without_progress_threshold(1500);
                if (trainer_c.get_learning_rate() >= trainer_c.get_min_learning_rate()) {
                    while (trainer_c.get_learning_rate() >= trainer_c.get_min_learning_rate() && !g_interrupt_signal_received) {
                        if (data.generate_samples(mini_batch_size, samples, labels, false)) trainer_c.train_one_step(samples, labels);
                        else g_interrupt_signal_received = true;
                    }
                    trainer_c.get_net();
                    net_b.clean();
                    dlib::serialize("llm_shakespeare_model_a.dat") << net_b;
                    cout << "shakespeare model saved: llm_shakespeare_model_a.dat" << endl;
                    cout << "shakespeare model parameters: " << count_parameters(net_b) << endl;
                    g_interrupt_signal_received = false;

                    // Test the network with the same data to ensure it has learned something
                    std::vector<unsigned long> predicted_labels_c = net_b(samples_txt);
                    size_t num_correct_c = 0;
                    for (size_t i = 0; i < labels_txt.size(); ++i) if (predicted_labels_c[i] == labels_txt[i]) ++num_correct_c;
                    double accuracy_c = static_cast<double>(num_correct_c) / labels_txt.size();
                    DLIB_TEST_MSG(accuracy_c > 0.9, "shakespeare model (accuracy: " + to_string(accuracy_c) + ") - right: " +\
                        to_string(num_correct_c) + " - wrong: " + to_string(labels_txt.size() - num_correct_c));
                }
                // Predict the next sequence of characters
                string input_sequence = "HAMLET By William Shakespeare - Act Three, Scene One\nTo be or not to be—that is the quest";
                std::vector<matrix<int, 0, 1>> input_tokens = tokenize_text(input_sequence, llm::sequence_size);
                string start_seq = to_unsigned_char_string(input_tokens.back());
                size_t pos = input_sequence.find(start_seq);
                if (pos != string::npos) input_sequence = input_sequence.substr(0, pos + start_seq.length());
                cout << "input sequence for text generation: <" << start_seq << ">" << endl;
                matrix<int> next_input(llm::sequence_size, 1);
                for (int i = 0; i < 450; ++i) {
                    unsigned long next_char = net_b(input_tokens.back());
                    input_sequence += static_cast<unsigned char>(next_char);

                    for (int j = 0; j < (llm::sequence_size - 1); ++j) next_input(j, 0) = input_tokens.back()(j + 1, 0);
                    int insert_pos = std::distance(
                        input_tokens.back().begin(),
                        std::find_if(input_tokens.back().begin(), input_tokens.back().end(),
                            [&](const auto& element) { return element == pad_id; })
                    );
                    if (insert_pos == llm::sequence_size) insert_pos = llm::sequence_size - 1;
                    next_input(insert_pos, 0) = static_cast<int>(next_char);

                    input_tokens.clear();
                    input_tokens.push_back(next_input);
                }
                cout << "generated text:\n\n" << input_sequence << " (...)\n\n";

                // Loading now the complete Shakespeare file
                string shakespeare_file = "shakespeare.txt";
                if (fs::exists(shakespeare_file)) {
                    documents shakespeare_data(llm::sequence_size, 0, true);
                    shakespeare_data.load_documents(shakespeare_file, false);
                    cout << "loading about " << shakespeare_data.get_total_samples() << " samples from " << shakespeare_file << endl;

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
                        cout << "restarting from the last checkpoint" << endl;
                    }

                    dnn_trainer<net_type_b> trainer_d(net_b, sgd(weight_decay, beta1), gpus);
                    trainer_d.set_learning_rate(learning_rate);
                    trainer_d.set_min_learning_rate(min_learning_rate);
                    trainer_d.set_mini_batch_size(mini_batch_size);
                    trainer_d.be_verbose();                    
                    trainer_d.set_synchronization_file("llm_shakespeare_model_b.ckp", std::chrono::minutes(5));
                    trainer_d.set_iterations_without_progress_threshold(iterations_without_progress_threshold);

                    // New training loop
                    while (trainer_d.get_learning_rate() >= trainer_d.get_min_learning_rate() && !g_interrupt_signal_received) {
                        if (shakespeare_data.generate_samples(mini_batch_size, samples, labels, false)) trainer_d.train_one_step(samples, labels);                        
                        else g_interrupt_signal_received = true;
                    }
                    trainer_d.get_net();
                    net_b.clean();
                    serialize("llm_shakespeare_model_b.dat") << net_b;
                    cout << "advanced shakespeare model saved: llm_shakespeare_model_b.dat" << endl;
                    cout << "advanced shakespeare model parameters: " << count_parameters(net_b) << endl;

                    // Attempting to generate a new sonnet
                    string sonnet_start = "Shall I compare thee to a winter's night?\nThy beauty warms the frost - bitten bough.\nIn darkness, thou art my guiding light,\nA beacon of hope 'midst winter's vow.";
                    std::vector<matrix<int, 0, 1>> input_tokens = tokenize_text(sonnet_start+"\n", llm::sequence_size);
                    if (!input_tokens.empty()) {
                        string generated_sonnet;
                        matrix<int> next_input(llm::sequence_size, 1);

                        cout << "generated sonnet:\n\n";
                        for (int i = 0; i < 700 && !input_tokens.empty(); ++i) {
                            unsigned long next_char = net_b(input_tokens.back());
                            unsigned char c = static_cast<unsigned char>(next_char);
                            generated_sonnet += c;

                            for (int j = 0; j < (llm::sequence_size - 1); ++j) next_input(j, 0) = input_tokens.back()(j + 1, 0);
                            int insert_pos = std::distance(
                                input_tokens.back().begin(),
                                std::find_if(input_tokens.back().begin(), input_tokens.back().end(),
                                    [&](const auto& element) { return element == pad_id; })
                            );
                            if (insert_pos == llm::sequence_size) insert_pos = llm::sequence_size - 1;
                            next_input(insert_pos, 0) = static_cast<int>(next_char);

                            input_tokens.clear();
                            input_tokens.push_back(next_input);

                            // Stop after generating what looks like a complete sonnet
                            if (generated_sonnet.find("END") != string::npos || generated_sonnet.find("\n\n") != string::npos) break;
                        }
                        cout << sonnet_start << "\n" << generated_sonnet << "\n";

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
        llm::net_v1_1 net;
        softmax<multiply<llm::net_v1_1::subnet_type>> generator(multiply_(1.0 / temperature));
        if (fs::exists(language_model)) deserialize(language_model) >> net;
        else {
            cerr << "language model not found! (<" << language_model << ">)" << endl;
            return 1;
        }
        generator.subnet().subnet() = net.subnet();
        cout << "number of model parameters: " << count_parameters(generator) << endl << endl;
        context_window prompt(llm::sequence_size);
        string input = "The salesperson", output = "";
        cout << "Input prompt: " << input << " (...)" << endl;
        cout << "Generated text: " << input << " ";

        std::vector<int> prompt_ids, endings = { eos_id, pad_id }, response_ids;
        sp.Encode(dlib::trim(input), &prompt_ids);
        prompt.add_input(prompt_ids);
        matrix<int, 0, 1> padded_window;
        for (int i = 0; i < 100; ++i) {
            if (prompt.get_padded_window(padded_window)) {
                matrix<float, llm::vocab_size, 1> logits = mat(generator(padded_window));
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
        std::vector<int> vocab_sizes = { 3000, 8000, 12000, 20000, 40000, 80000, 100000 };
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
            else if (vocab_size == 12000) size_suffix = "12k";
            else if (vocab_size == 20000) size_suffix = "20k";
            else if (vocab_size == 40000) size_suffix = "40k";
            else if (vocab_size == 80000) size_suffix = "80k";
            else if (vocab_size == 100000) size_suffix = "100k";
            string current_vocabulary_prefix = "ernie.eu.ung." + size_suffix;
            //string current_vocabulary_prefix = "ernie.en-fr.ung." + size_suffix;

            string train_args = "--input=" + corpus_files +
                " --model_prefix=" + current_vocabulary_prefix +
                " --bos_id=" + to_string(bos_id) + " --eos_id=" + to_string(eos_id) +
                " --unk_id=" + to_string(unk_id) + " --pad_id=" + to_string(pad_id) +
                " --model_type=unigram" +
                " --character_coverage=1.0" +
                " --max_sentence_length=16768" +
                " --split_by_unicode_script=false" +
                " --input_sentence_size=30000000" +
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
        llm::net_v1_1 net;
        adam solver(weight_decay, beta1, beta2);
        dnn_trainer<llm::net_v1_1, adam> my_trainer(net, solver, gpus);
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
        cout << "about " << data.get_total_samples() << " samples for the training" << endl;

        // Training loop
        while (!g_interrupt_signal_received && my_trainer.get_learning_rate() >= my_trainer.get_min_learning_rate()) {
            if (data.generate_samples(mini_batch_size, samples, labels)) my_trainer.train_one_step(samples, labels);
            else g_interrupt_signal_received = true;            
        }
        cout << "stopping the training process" << endl;
        my_trainer.get_net();
        net.clean();
        serialize(language_model) << net;
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
            cout << "input vector: ";
            std::copy(input.begin(), input.end(), std::ostream_iterator<int>(cout, " "));
            cout << endl;
            prompt.add_input(input);
            matrix<int, 0, 1> padded_window;
            for (size_t i = 0; i < 60; i++) {
                if (prompt.get_padded_window(padded_window)) {
                    cout << "padded window (i=" << i << "): " << padded_window;
                    cout << endl;
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
        llm::net_v1_1 net;
        softmax<multiply<llm::net_v1_1::subnet_type>> generator(multiply_(1.0 / temperature));
        if (fs::exists(language_model)) deserialize(language_model) >> net;
        else {
            cerr << "language model not found! (<" << language_model << ">)" << endl;
            return 1;
        }
        generator.subnet().subnet() = net.subnet();
        cout << "number of model parameters: " << count_parameters(generator) << endl;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(llm::sequence_size/2, llm::sequence_size*6);
        context_window prompt(llm::sequence_size);
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
                size_t total_steps = std::min(static_cast<int>(llm::sequence_size), dis(gen));
                // Generate response
                response_ids.clear();
                cout << "[ERNIE] ";
                matrix<int, 0, 1> padded_window;
                int cur_top_k = (top_k >= 1) ? top_k : 1, predicted_label;
                for (int i = 0; i < total_steps; ++i) {                    
                    if (prompt.get_padded_window(padded_window)) {
                        //cout << padded_window << endl;
                        matrix<float, llm::vocab_size, 1> logits = mat(generator(padded_window));
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