/*
    This is an example illustrating the implementation of a Very Small Language Model (VSLM)
    using the deep learning tools from the dlib C++ Library. The program, named ERNIE
    (Efficient Rapid Neural Intelligence Engine), demonstrates how to extend dlib's
    capabilities to handle natural language processing tasks, specifically focusing on
    transformer-based architectures.

    Key features of this implementation include:
    - Custom layers designed for matrix-based processing of inputs, optimized for dlib's
      tensor structure.
    - Specialized input layers for SLM, including embedding injection and positional encoding.
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

#include "slm_defs.h"
#include "advanced_tokenizer.hpp"
#include "data.h"

#include <sentencepiece_trainer.h>
#include <sentencepiece_processor.h>

namespace fs = std::filesystem;
namespace po = boost::program_options;
using namespace dlib;

volatile std::sig_atomic_t g_interrupt_signal_received = 0;
void signalHandler(int signal) {
    if (signal == SIGINT) {
        g_interrupt_signal_received = 1;
        cout << "\ninterrupt detected (CTRL+C), cleaning up and closing the program" << endl;
    }
}

const int bos_id = 0, eos_id = 1, unk_id = 2, pad_id = 3;
struct a_training {
    std::vector<matrix<int, 0, 1>> samples;
    std::vector<unsigned long> labels;
};

// Other global parameters
string vocabulary_prefix = "ernie.en-fr.ung.50k", language_model = "ernie_fp32_v1.dat";
std::unique_ptr<advanced_tokenizer> tokenizer_;

void configure_console() {
    SetConsoleOutputCP(CP_UTF8);
    int res = _setmode(_fileno(stdout), _O_TEXT);
    if (res == -1) cerr << "Cannot set mode" << endl;
    std::cout.imbue(std::locale("en_US.UTF-8"));    
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
                std::cout << "parsing file: " << entry.path().string() << endl;
                output << input.rdbuf() << "\n";
            }
        }
    }

}
using utils::replace_html_entities;
using utils::is_utf8;
using utils::concatenate_files;

const size_t std_global_context_size = (5 * transformer::vslm::MAX_SEQ_LEN);
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
    documents(size_t seq_size = transformer::vslm::MAX_SEQ_LEN, int pad_value = pad_id,
        bool use_letter_tokenization = false, long token_limit = -1) :
        sequence_size_(seq_size), pad_value_(pad_value),
        use_letter_tokenization_(use_letter_tokenization), token_limit_(token_limit) {
        is_initialized_ = false;
        if (!use_letter_tokenization_) {
            if (fs::exists(vocabulary_prefix + ".model")) {
                auto status = sp_.Load(vocabulary_prefix + ".model");
                if (!status.ok()) cerr << "error loading SentencePiece model: " << status.ToString() << endl;
                else is_initialized_ = true;                
            } else cerr << "vocabulary file not found! (<" << (vocabulary_prefix + ".model|.vocab") << ">)" << endl;
        } else is_initialized_ = true;
        clear_all();
    }
    
    void set_use_letter_tokenization(bool use_letter_tokenization) {
        if (use_letter_tokenization_ != use_letter_tokenization) {            
            is_initialized_ = false;
            use_letter_tokenization_ = use_letter_tokenization;
            if (!use_letter_tokenization_) {
                if (fs::exists(vocabulary_prefix + ".model")) {
                    auto status = sp_.Load(vocabulary_prefix + ".model");
                    if (!status.ok()) cerr << "error loading SentencePiece model: " << status.ToString() << endl;
                    else is_initialized_ = true;
                }
                else cerr << "vocabulary file not found! (<" << (vocabulary_prefix + ".model|.vocab") << ">)" << endl;
            } else is_initialized_ = true;
            clear_all();
        }
    }
    void set_samples_idx(size_t v) { 
        const std::lock_guard<std::mutex> lock(g_mutex_); 
        samples_idx_ = v;
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

            if (token_limit_ == -1 || (current_total_tokens + tokens.size()) < token_limit_) {
                new_tokens.push_back(tokens);
                current_total_tokens += tokens.size();
            } else {
                size_t remaining_space = token_limit_ - current_total_tokens;
                if (remaining_space > 0) {
                    std::vector<int> truncated_tokens(tokens.begin(), tokens.begin() + remaining_space);
                    new_tokens.push_back(truncated_tokens);
                    current_total_tokens = token_limit_;
                }
            }
        };

        if (split_sentences) {
            std::vector<string> sentences = split_into_sentences(text);
            for (const auto& sentence : sentences) {
                if (token_limit_ != -1 && current_total_tokens >= token_limit_) break;
                std::vector<int> tokens = preprocess_sentence(sentence);
                process_tokens(tokens);
            }
        } else {
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
                std::cout << "loading file: " << fs_path.string() << endl;
                std::ifstream file(fs_path, std::ios::binary);
                string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
                if (!is_utf8(content)) std::cout << "warning - file <" << fs_path.string() << "> seems not to be UTF-8 encoded" << endl;
                load_text(content, split_sentences);
            } else if (fs::is_directory(fs_path)) {
                for (const auto& entry : fs::recursive_directory_iterator(fs_path)) {
                    if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                        std::cout << "loading file: " << entry.path().string() << endl;
                        std::ifstream file(entry.path(), std::ios::binary);
                        string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
                        if (!is_utf8(content)) std::cout << "warning - file <" << entry.path().string() << "> seems not to be UTF-8 encoded" << endl;
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

        static std::random_device rd;
        static std::mt19937 gen(rd());
        if (select_randomly) {
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
            const std::lock_guard<std::mutex> lock(g_mutex_);
            if (pre_samples_.size() == 0) {
                samples_idx_ = 0;
                for (const auto& sentence : source_tokens_) {                    
                    if (sentence.size() > (sequence_size_ + 1)) {
                        for (size_t i = 0; i < (sentence.size() - (sequence_size_ + 1)); ++i) {
                            matrix<int> sample(sequence_size_, 1);
                            for (size_t j = 0; j < sequence_size_; ++j) sample(j, 0) = sentence[i + j];
                            pre_samples_.push_back(sample);
                            pre_labels_.push_back(static_cast<unsigned long>(sentence[i + sequence_size_]));
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

    string to_unsigned_char_string(const std::vector<int>& ints) const {
        string result;
        if (use_letter_tokenization_) {
            for (int v = 0; v < ints.size(); ++v) {
                if (ints[v] > static_cast<int>(std::numeric_limits<unsigned char>::max()))
                    cerr << "Value too large for 'unsigned char' conversion (" << to_string(ints[v]) << ")" << endl;
                result += static_cast<unsigned char>(ints[v]);
            }
        } else {
            sp_.Decode(ints, &result);
        }
        return result;
    }
    string to_unsigned_char_string(const matrix<int, 0, 1>& ints) const {        
        string result;
        if (use_letter_tokenization_) {
            for (int v = 0; v < ints.nr(); ++v) {
                if (ints(v, 0) > static_cast<int>(std::numeric_limits<unsigned char>::max()))
                    cerr << "Value too large for 'unsigned char' conversion (" << to_string(ints(v, 0)) << ")" << endl;
                result += static_cast<unsigned char>(ints(v, 0));
            }
        } else {
            std::vector<int> tokens(ints.nr());
            for (long v = 0; v < ints.nr(); ++v) tokens[v] = ints(v, 0);
            sp_.Decode(tokens, &result);            
        }
        return result;
    }
    string to_unsigned_char_string(unsigned long value) const {
        if (use_letter_tokenization_) {
            if (value > static_cast<unsigned long>(std::numeric_limits<unsigned char>::max()))
                cerr << "Value too large for 'unsigned char' conversion (" << to_string(value) << ")" << endl;
            return string(1, static_cast<unsigned char>(value));
        } else {
            if (value > static_cast<unsigned long>(std::numeric_limits<int>::max()))
                cerr << "Value too large for 'int' conversion (" << to_string(value) << ")" << endl;
            string result;
            sp_.Decode({ static_cast<int>(value) }, &result);
            return result;
        }
    }

private:
    size_t sequence_size_;
    std::vector<std::vector<int>> source_tokens_;
    std::vector<matrix<int, 0, 1>> pre_samples_;
    std::vector<unsigned long> pre_labels_;
    size_t samples_idx_;
    std::mutex g_mutex_;

    sentencepiece::SentencePieceProcessor sp_;
    long total_tokens_, token_limit_;
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
            for (size_t i = 0; i < cleaned_sentence.size(); ++i) tokens.push_back(static_cast<int>(cleaned_sentence.at(i)));
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
    MY_TEST_MSG(max(abs(mat(output_cpu_b) - mat(input))) < 1e-5,
        "transpose_cpu: max(abs(mat(output_cpu_b) - mat(input))) < 1e-5");

#ifdef DLIB_USE_CUDA
    input /= 2;
    resizable_tensor output_cuda_a, output_cuda_b(input);
    output_cuda_a.copy_size(output_cpu_a);
    cuda::transpose(false, output_cuda_a, input);
    cuda::transpose(true, output_cuda_b, output_cuda_a);
    MY_TEST_MSG(max(abs(mat(output_cpu_a) - mat(output_cuda_a))) < 1e-5,
        "transpose_cuda: max(abs(mat(output_cpu_a) - mat(output_cuda_a))) < 1e-5");
    MY_TEST_MSG(max(abs(mat(output_cpu_b) - mat(output_cuda_b))) < 1e-5,
        "transpose_cuda: max(abs(mat(output_cpu_b) - mat(output_cuda_b))) < 1e-5");
#endif
}

void test_hsplit_hstack() {
    const long num_heads = 4;
    const long num_samples = 1;
    const long input_k = 1;
    const long input_nr = 8;
    const long input_nc = 12;

    using net_type = tag1<hstack<hsplit<num_heads, dlib::input<matrix<float>>>>>;
    net_type net;

    resizable_tensor input_tensor;
    input_tensor.set_size(num_samples, input_k, input_nr, input_nc);
    tt::tensor_rand rnd(std::rand());
    rnd.fill_uniform(input_tensor);

    net.forward(input_tensor);
    auto& output_tensor = layer<tag1>(net).get_output();

    MY_TEST_MSG(output_tensor.num_samples() == input_tensor.num_samples(),
        "hsplit_hstack: output_tensor.num_samples() == input_tensor.num_samples()");
    MY_TEST_MSG(output_tensor.k() == input_tensor.k(),
        "hsplit_hstack: output_tensor.k() == input_tensor.k()");
    MY_TEST_MSG(output_tensor.nr() == input_tensor.nr(),
        "hsplit_hstack: output_tensor.nr() == input_tensor.nr()");
    MY_TEST_MSG(output_tensor.nc() == input_tensor.nc(),
        "hsplit_hstack: output_tensor.nc() == input_tensor.nc()");
    MY_TEST_MSG(max(abs(mat(output_tensor) - mat(input_tensor))) < 1e-5,
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
    MY_TEST_MSG(max(abs(mat(grad_cpu) - mat(input))) < 1e-5,
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
    MY_TEST_MSG(max(abs(mat(net_output) - expected_output)) < 1e-5, "positional_encodings layer");
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
    MY_TEST_MSG(acc > 0.9, "embeddings accuracy: " + to_string(acc));
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
    MY_TEST_MSG(!error_found, "Normalized values vs expected values");

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
    MY_TEST_MSG(!backward_error_found, "Backward pass values vs expected values");

#ifdef DLIB_USE_CUDA
    resizable_tensor y_cuda(x);
    resizable_tensor scale_cuda;
    cuda::rms_normalize(eps, y_cuda, scale_cuda, x, gamma);
    MY_TEST_MSG(max(abs(mat(y_cpu) - mat(y_cuda))) < 1e-5, "max(abs(mat(y_cpu) - mat(y_cuda))) < 1e-5");
    MY_TEST_MSG(max(abs(mat(scale_cpu) - mat(scale_cuda))) < 1e-5,
        "max(abs(mat(scale_cpu) - mat(scale_cuda))) < 1e-5");

    resizable_tensor src_grad_cuda(x), gamma_grad_cuda(1, x.k());
    resizable_tensor dscale_cuda(x.num_samples());
    src_grad_cuda = 0;
    cuda::rms_normalize_gradient(gradient_input, scale_cuda, x, gamma, src_grad_cuda, gamma_grad_cuda, dscale_cuda);
    MY_TEST_MSG(max(abs(mat(src_grad_cpu) - mat(src_grad_cuda))) < 1e-5,
        "max(abs(mat(src_grad_cpu) - mat(src_grad_cuda))) < 1e-5");
    MY_TEST_MSG(max(abs(mat(gamma_grad_cpu) - mat(gamma_grad_cuda))) < 1e-5,
        "max(abs(mat(gamma_grad_cpu) - mat(gamma_grad_cuda))) < 1e-5");
    MY_TEST_MSG(max(abs(mat(dscale_cpu) - mat(dscale_cuda))) < 1e-5,
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
    MY_TEST_MSG(max(abs(mat(net_output) - mat(expected_output))) < 1e-5, "tril layer");
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
    MY_TEST_MSG(max(abs(mat(net_output) - mat(expected_output))) < 1e-5, "multm_prev layer");
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
    MY_TEST_MSG(max(abs(mat(net_output) - mat(expected_output))) < 1e-5, "softmaxm layer");

    // Compare CPU and CUDA utility functions
    resizable_tensor output_tensor, cpu_grad, gradient_input;
    output_tensor.copy_size(input_tensor);
    cpu_grad.copy_size(input_tensor);
    cpu_grad = 0;
    gradient_input.copy_size(input_tensor);    
    randomize_parameters(gradient_input, nr + nc, rnd);
    cpu::softmax(output_tensor, input_tensor, tt::operation_mode::PLANE_WISE);    
    cpu::softmax_gradient(cpu_grad, output_tensor, gradient_input, tt::operation_mode::PLANE_WISE);
    MY_TEST_MSG(max(abs(mat(output_tensor) - mat(expected_output))) < 1e-5, "softmax (cpu)");
#ifdef DLIB_USE_CUDA
    resizable_tensor cuda_grad;
    cuda_grad.copy_size(input_tensor);
    cuda_grad = 0;
    cuda::softmax(output_tensor, input_tensor, tt::operation_mode::PLANE_WISE);
    cpu::softmax_gradient(cuda_grad, output_tensor, gradient_input, tt::operation_mode::PLANE_WISE);
    MY_TEST_MSG(max(abs(mat(output_tensor) - mat(expected_output))) < 1e-5, "softmax (cuda)");
    MY_TEST_MSG(max(abs(mat(cuda_grad) - mat(cpu_grad))) < 1e-5, "softmax_gradient cpu-cuda");
#endif
}

void test_linear()
{
    // Define the network
    using net_type = tag2<linear_no_bias<6, tag1<input<matrix<float>>>>>;
    net_type net;

    // Input tensor
    const int n_samples = 3, k = 1;
    std::vector<matrix<float>> x(n_samples);
    matrix<float> xtmp(2, 4);
    xtmp = 1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f;
    x[0] = xtmp;
    xtmp = 9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f;
    x[1] = xtmp;
    xtmp = 17.0f, 18.0f, 19.0f, 20.0f,
        21.0f, 22.0f, 23.0f, 24.0f;
    x[2] = xtmp;

    // Convert input matrix to tensor
    resizable_tensor input_tensor;
    net.to_tensor(&x[0], &x[0] + n_samples, input_tensor);
    net.forward(input_tensor);

    // Get the internal linear weights
    matrix<float> w = mat(layer<tag2>(net).subnet().layer_details().get_weights());

    // Theoretical calculation of the output
    std::vector<matrix<float>> expected_outputs(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        matrix<float> input_matrix = x[i];
        expected_outputs[i] = input_matrix * w;
    }

    // Compare output tensor with expected output
    auto& net_output = layer<tag2>(net).get_output();

    // Display results
    for (int i = 0; i < n_samples; ++i) {
        matrix<float> output_sample;
        output_sample.set_size(2, 6);
        for (long r = 0; r < output_sample.nr(); ++r) {
            for (long c = 0; c < output_sample.nc(); ++c) {
                output_sample(r, c) = net_output.host()[tensor_index(net_output, i, 0, r, c)];
            }
        }
        MY_TEST_MSG(max(abs(output_sample - expected_outputs[i])) < 1e-5,
            "linear layer - sample " + std::to_string(i));
    } 
}

void test_loss_cross_entropy()
{
    //print_spinner();
    constexpr int input_height = 5;
    constexpr int input_width = 3;
    const size_t num_samples = 50;
    const size_t num_classes = 4;

    std::vector<matrix<double>> x(num_samples);
    std::vector<unsigned long> y(num_samples);
    matrix<double> xtmp(input_height, input_width);

    dlib::rand rnd;
    for (size_t ii = 0; ii < num_samples; ++ii)
    {
        for (int jj = 0; jj < input_height; ++jj)
            for (int kk = 0; kk < input_width; ++kk)
                xtmp(jj, kk) = rnd.get_random_float();
        x[ii] = xtmp;
        y[ii] = rnd.get_integer_in_range(0, num_classes);
    }

    using net_type = loss_cross_entropy<linear_no_bias<num_classes, input<matrix<double>>>>;

    net_type net;
    dnn_trainer<net_type> trainer(net, sgd(0, 0.9));
    trainer.set_learning_rate(0.1);
    trainer.set_min_learning_rate(0.01);
    trainer.set_mini_batch_size(10);
    trainer.set_max_num_epochs(100);
    trainer.train(x, y);

    const std::vector<unsigned long> predictions = net(x);
    int correct_predictions = 0, incorrect_predictions = 0;
    for (size_t ii = 0; ii < num_samples; ++ii)
    {
        if (predictions[ii] == y[ii])
            ++correct_predictions;
        else
            ++incorrect_predictions;
    }

    MY_TEST_MSG(correct_predictions > incorrect_predictions,
        "Predicted labels (" << correct_predictions << ") do not dominate: correct="
        << correct_predictions << ", incorrect=" << incorrect_predictions);
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
    std::cout << endl <<
        "███████╗██████╗ ███╗   ██╗██╗███████╗\n"
        "██╔════╝██╔══██╗████╗  ██║██║██╔════╝\n"
        "█████╗  ██████╔╝██╔██╗ ██║██║█████╗  \n"
        "██╔══╝  ██╔══██╗██║╚██╗██║██║██╔══╝  \n"
        "███████╗██║  ██║██║ ╚████║██║███████╗\n"
        "╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚══════╝\n"
        "Welcome to the ERNIE generative AI program! (version 1.1.6)\n\n";
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
            std::cout << oss.str() << endl;
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
            true,      // 1: documents class
            true,      // 2: transpose layer
            true,      // 3: tril layer
            true,      // 4: positional_encodings layer
            true,      // 5: embeddings layer
            true,      // 6: multm_prev layer
            true,      // 7: softmax layer
            true,      // 8: attention mechanism
            true,      // 9: linear layer       
            true,      // 10: hsplit/hstack layers
            true,      // 11: rms_norm layer
            true,      // 12: loss_cross_entropy loss
            true,      // 13: multihead attention model
            false      // 14: "shakespeare" example
        };

        // test: tokenization
        if (!skip_tests[0]) {
            if (display_debug_info) std::cout << "test: strings & tokenization\n";
            string sentence = "  &nbsp;&lt;p&gt;Hellooooooo     frieeeends !!!!!! This is sooooooo coooool &amp; awesoooooome !&lt;/p&gt;  ";
            string cleaned_sentence = dlib::trim(replace_html_entities(std::regex_replace(sentence, std::regex("(.)\\1{4,}"), "$1$1$1$1")));
            std::cout << "string normalisation: [" << sentence << "] => [" << cleaned_sentence << "]" << endl;

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
                std::cout << "sentence: " << sentence << endl << "Tokens: ";
                for (const auto& token : tokens) std::cout << token << " ";

                std::vector<int> ids;
                sp.Encode(sentence, &ids);
                std::cout << endl << "token IDs: ";
                for (const auto& id : ids) std::cout << id << " ";

                string recovered_text;
                sp.Decode(ids, &recovered_text);
                std::cout << endl << "original sentence : " << recovered_text << endl << endl;
            }
        }

        // test: documents
        if (!skip_tests[1]) {
            if (display_debug_info) std::cout << "test: documents class\n";
            {                
                string shakespeare_file = "shakespeare.txt", combined_str;
                if (fs::exists(shakespeare_file)) {
                    std::ifstream file(shakespeare_file);
                    string source_text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

                    long found_count = 0, total_tests = 100;
                    std::vector<matrix<int, 0, 1>> samples;
                    std::vector<unsigned long> labels;
                    documents data(15, pad_id, true);
                    data.load_documents(shakespeare_file, false);
                    for (int i = 0; i < total_tests; ++i) {
                        data.generate_samples(1, samples, labels);
                        if (!samples.empty() && !labels.empty()) {
                            std::vector<int> tokens(samples[0].nr() + 1);
                            for (long i = 0; i < samples[0].nr(); ++i) tokens[i] = samples[0](i, 0);
                            tokens[tokens.size() - 1] = labels[0];
                            combined_str = data.to_unsigned_char_string(tokens);
                            if (source_text.find(combined_str) != string::npos) {
                                if (display_debug_info && found_count < 5) std::cout << "(" << combined_str << ")" << endl;
                                found_count++;
                            } else {
                                if (display_debug_info) std::cout << "error: (" << combined_str << ")" << endl;
                            }
                        }
                        samples.clear();
                        labels.clear();
                    }
                    std::cout << "sequences found in source text: " << found_count << " out of " << total_tests << endl;
                    MY_TEST_MSG(static_cast<double>(found_count) / total_tests > 0.9, "documents class test_0");
                    data.clear_all();

                    data.set_use_letter_tokenization(false);
                    data.load_documents(shakespeare_file, false);
                    source_text = std::regex_replace(source_text, std::regex("\n{2,}"), "\n");
                    source_text = std::regex_replace(source_text, std::regex("\n"), " ");
                    found_count = 0;
                    for (int i = 0; i < total_tests; ++i) {
                        data.generate_samples(1, samples, labels);
                        if (!samples.empty() && !labels.empty()) {
                            std::vector<int> tokens(samples[0].nr() + 1);
                            for (long i = 0; i < samples[0].nr(); ++i) tokens[i] = samples[0](i, 0);
                            tokens[tokens.size() - 1] = labels[0];
                            combined_str = data.to_unsigned_char_string(tokens);
                            if (source_text.find(combined_str) != string::npos) {
                                if (display_debug_info && found_count < 5) std::cout << "(" << combined_str << ")" << endl;
                                found_count++;
                            } else {
                                if (display_debug_info) std::cout << "error: (" << combined_str << ")" << endl;
                            }
                        }
                        samples.clear();
                        labels.clear();
                    }
                    std::cout << "sequences found in source text: " << found_count << " out of " << total_tests << endl;
                    MY_TEST_MSG(static_cast<double>(found_count) / total_tests > 0.9, "documents class test_1");
                    data.clear_all();
                }
            }
        }

        // test: transpose layer
        if (!skip_tests[2]) {
            if (display_debug_info) cout << "test: transpose layer\n";
            test_transpose();
            {
                transpose_ l;
                auto res = test_layer(l);
                MY_TEST_MSG(res, " transpose test_0 layer\n" + res);
            }
        }        

        // test: tril layer
        if (!skip_tests[3]) {
            if (display_debug_info) cout << "\ntest: tril layer\n";
            test_tril();
            {
                tril_<0, neg_infinity_tag> l;
                auto res = test_layer(l);
                MY_TEST_MSG(res, " tril test_0 layer\n" + res);
            }
            {
                tril_<3, zero_tag> l;
                auto res = test_layer(l);
                MY_TEST_MSG(res, " tril test_1 layer\n" + res);
            }
            {
                tril_<-5, void, 1, 2> l;
                auto res = test_layer(l);
                MY_TEST_MSG(res, " tril test_2 layer\n" + res);
            }
        }

        // test: positional_encodings layer
        if (!skip_tests[4]) {
            if (display_debug_info) cout << "test: positional_encodings layer\n";
            test_positional_encodings();
            {
                positional_encodings_ l;
                auto res = test_layer(l);
                MY_TEST_MSG(res, " positional_encodings test layer\n" + res);
            }
        }

        // test: embeddings layer
        if (!skip_tests[5]) {
            if (display_debug_info) cout << "test: embeddings layer\n";
            test_embeddings();
            {
                embeddings_<7, 12> l;
                auto res = test_layer(l);
                MY_TEST_MSG(res, " embeddings test layer\n" + res);
            }
        }

        // test: multm_prev layer
        if (!skip_tests[6]) {
            if (display_debug_info) cout << "test: multm_prev layer\n";
            test_multm_prev();            
        }

        // test: softmax layer
        if (!skip_tests[7]) {
            if (display_debug_info) cout << "\ntest: softmax layer\n";
            test_softmaxm();
            {
                softmax2_<static_cast<unsigned long>(tt::operation_mode::CHANNEL_WISE)> l;
                auto res = test_layer(l);
                MY_TEST_MSG(res, " softmaxm test_0 layer\n" + res);
            }
            {
                softmax2_<static_cast<unsigned long>(tt::operation_mode::PLANE_WISE)> l;
                auto res = test_layer(l);
                MY_TEST_MSG(res, " softmaxm test_1 layer\n" + res);
            }
        }

        // test: attention mechanism
        if (!skip_tests[8]) {
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
                using net_type = tag10<multm_prev1<tag7<softmaxm<transformer::scale_weights<3, tag6<multm_prev4<
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
                MY_TEST_MSG(max(abs(mat(net_output) - Z)) < 1e-5, "attention mechanism");
            }
        }

        // test: linear layer
        if (!skip_tests[9]) {
            if (display_debug_info) cout << "\ntest: linear layer\n";
            test_linear();
            {
                linear_<4, LINEAR_NO_BIAS> l;
                auto res = test_layer(l);
                MY_TEST_MSG(res, " linear test_0 layer\n" + res);
            }
            {
                linear_<3, LINEAR_HAS_BIAS> l;
                auto res = test_layer(l);
                MY_TEST_MSG(res, " linear test_1 layer\n" + res);
            }
        }

        // test: hsplit/hstack layers
        if (!skip_tests[10]) {
            if (display_debug_info) cout << "\ntest: hsplit/hstack layers\n";                            
            test_hsplit_hstack();
            {
                hstack_ l;
                auto res = test_layer(l);
                MY_TEST_MSG(res, " hsplit/hstack layers\n" + res);
            }
        }

        // test: rms_norm layer
        if (!skip_tests[11]) {
            if (display_debug_info) cout << "\ntest: rms_norm layer\n";
            {
                test_rms_normalize();
                {
                    rms_norm_ l;
                    auto res = test_layer(l);
                    MY_TEST_MSG(res, " RMS normalize layer" + res);
                }
            }
        }

        // test: rms_norm layer
        if (!skip_tests[12]) {
            if (display_debug_info) cout << "\ntest: loss_cross_entropy loss\n";
            test_loss_cross_entropy();
        }

        // test: multihead attention model
        if (!skip_tests[13]) {
            mini_batch_size = 32;
            if (display_debug_info) cout << "\ntest: multihead attention model\n";
            const long num_heads = 4;
            const long embedding_dim = 64;
            const long max_seq_len = 48;
            const long num_classes = 256;
            const long num_epochs = 3000;
            const long num_samples = (1000 / mini_batch_size) * mini_batch_size;

            // Generate synthetic training data
            dlib::rand rnd(std::rand());
            std::vector<matrix<int, 0, 1>> samples;
            std::vector<unsigned long> labels;
            for (int i = 0; i < num_samples; ++i) {
                matrix<int> sample(max_seq_len, 1);
                for (int r = 0; r < max_seq_len; ++r) sample(r, 0) = rnd.get_random_32bit_number() % num_classes;
                samples.push_back(sample);
                labels.push_back(rnd.get_random_32bit_number() % num_classes);
            }
            auto count_unique_classes = [&labels]() {
                std::unordered_set<unsigned long> unique_classes(labels.begin(), labels.end());
                return unique_classes.size();
            };

            // Split data into batches
            std::vector<std::vector<matrix<int, 0, 1>>> batches;
            std::vector<std::vector<unsigned long>> label_batches;
            for (int i = 0; i < num_samples; i += mini_batch_size) {
                std::vector<matrix<int, 0, 1>> batch_samples(samples.begin() + i, samples.begin() + i + mini_batch_size);
                std::vector<unsigned long> batch_labels(labels.begin() + i, labels.begin() + i + mini_batch_size);
                batches.push_back(batch_samples);
                label_batches.push_back(batch_labels);
            }

            using train_net_a = transformer::classification_head<true, gelu, num_classes, embedding_dim,
                transformer::def::transformer_block<gelu, transformer::dropout_10, max_seq_len, embedding_dim, num_heads,
                transformer::positional_embeddings<num_classes, embedding_dim,
                input<matrix<int, 0, 1>>>>>;

            cout << "sequence size: " << max_seq_len << endl;
            cout << "embedding size: " << embedding_dim << endl;
            cout << "number of embeddings: " << num_classes << endl;
            cout << "batch size: " << mini_batch_size << endl;
            cout << "number of samples: " << samples.size() << endl;
            cout << "number of unique classes: " << count_unique_classes() << std::endl;
            cout << "number of batches: " << batches.size() << "/" << label_batches.size() << endl;

            train_net_a net_a;
            dnn_trainer<train_net_a, adam> trainer_a(net_a, adam(weight_decay, beta1, beta2), gpus);
            trainer_a.set_learning_rate(learning_rate);
            trainer_a.set_min_learning_rate(min_learning_rate);
            trainer_a.set_mini_batch_size(mini_batch_size);
            trainer_a.set_learning_rate_shrink_factor(0.1);
            trainer_a.set_iterations_without_progress_threshold(1000);
            for (size_t epoch = 0, step = 0; epoch < num_epochs && !g_interrupt_signal_received; ++epoch) {
                for (size_t i = 0; i < batches.size() && !g_interrupt_signal_received; ++i) trainer_a.train_one_step(batches[i], label_batches[i]);
                step += batches.size();
                if (epoch % 100 == 0) cout << "epoch[MAX]#: " << (epoch + 1) << "[" << num_epochs << "] step#: " <<
                    step << " learning rate : " <<
                    trainer_a.get_learning_rate() << " average loss: " <<
                    trainer_a.get_average_loss() << " steps without progress: " <<
                    trainer_a.get_steps_without_progress() << endl;
                if (trainer_a.get_learning_rate() < trainer_a.get_min_learning_rate()) break;
            }
            trainer_a.get_net();
            net_a.clean();
            //---
            visit_computational_layers(net_a, [](dropout_& l) { l = dropout_(0.0f); });
            cout << "multihead attention model parameters: " << count_parameters(net_a) << endl;
            std::vector<unsigned long> predicted_labels_a = net_a(samples);
            int num_correct_a = 0;
            for (size_t i = 0; i < labels.size(); ++i) if (predicted_labels_a[i] == labels[i]) ++num_correct_a;
            double accuracy_a = static_cast<double>(num_correct_a) / labels.size();
            MY_TEST_MSG(accuracy_a > 0.9, "multihead attention model (accuracy: " + to_string(accuracy_a) + ") - right: " + \
                to_string(num_correct_a) + " - wrong: " + to_string(labels.size() - num_correct_a));
        }

        // test: "shakespeare" example
        if (!skip_tests[14])        
        {
            if (display_debug_info) cout << "\ntest: test: \"shakespeare\" example\n";                        
            {
                mini_batch_size = 64;
                const long num_layers = 2;
                const long num_heads = 4;
                const long embedding_dim = 64;
                const long max_seq_len = 48;
                const long num_classes = 256;
                const long num_epochs = 3000;
                using net_type = transformer::transformer_config<num_classes, num_layers, num_heads, embedding_dim, max_seq_len>;

                // Tokenize the Shakespeare text
                string input_sequence = shakespeare_test;
                std::vector<matrix<int, 0, 1>> samples;
                std::vector<unsigned long> labels;
                documents data_b(max_seq_len, pad_id, true);
                data_b.load_text(shakespeare_text, false);
                data_b.generate_samples(1, samples, labels, false);                
                size_t num_samples = (data_b.get_total_presamples() / mini_batch_size) * mini_batch_size, step = 0;
                size_t num_batches = num_samples / mini_batch_size;                
                // Display SLM parameters
                cout << net_type::model_info::describe() << endl;
                cout << "batch size: " << mini_batch_size << endl;
                cout << "number of generated samples: " << num_samples << endl;
                cout << "number of batches: " << num_batches << endl;

                // Split data into batches
                data_b.set_samples_idx(0);
                data_b.generate_samples(num_samples, samples, labels, false);
                std::vector<std::vector<matrix<int, 0, 1>>> batches;
                std::vector<std::vector<unsigned long>> label_batches;
                for (int i = 0; i < num_samples; i += mini_batch_size) {
                    std::vector<matrix<int, 0, 1>> batch_samples(samples.begin() + i, samples.begin() + i + mini_batch_size);
                    std::vector<unsigned long> batch_labels(labels.begin() + i, labels.begin() + i + mini_batch_size);
                    batches.push_back(batch_samples);
                    label_batches.push_back(batch_labels);
                }

                // Use some threads to preload samples
                a_training a_training_sample;
                auto f = [&, mini_batch_size](documents& docs, dlib::pipe<a_training>& data, bool select_randomly) {
                    a_training temp;
                    while (data.is_enabled()) {
                        if (docs.generate_samples(mini_batch_size, temp.samples, temp.labels, select_randomly)) data.enqueue(temp);
                    }
                };                

                if (!fs::exists("slm_shakespeare_fp32_pre_v1.dat")) {
                    using train_net = net_type::network_type<true>;
                    train_net net_b;
                    dnn_trainer<train_net, adam> trainer_b(net_b, adam(weight_decay, beta1, beta2), gpus);
                    trainer_b.set_learning_rate(learning_rate);
                    trainer_b.set_min_learning_rate(min_learning_rate);
                    trainer_b.set_learning_rate_shrink_factor(0.1);
                    trainer_b.set_mini_batch_size(mini_batch_size);
                    trainer_b.set_iterations_without_progress_threshold(5000);                    

                    dlib::pipe<a_training> p_data(10);
                    data_b.set_samples_idx(0);
                    std::thread data_loader1([&data_b, &p_data, f]() { f(data_b, p_data, false); });
                    std::thread data_loader2([&data_b, &p_data, f]() { f(data_b, p_data, false); });
                    cout << "waiting for the initial pipe loading... ";
                    while (p_data.size() < 10) std::this_thread::sleep_for(std::chrono::seconds(1));
                    cout << "done" << endl;
                    
                    size_t num_epochs = 1500, epoch, b;
                    dlib::rand rnd(std::rand());
                    for (epoch = 0; epoch < num_epochs && !g_interrupt_signal_received; ++epoch) {
                        for (b = 0; b < num_batches && !g_interrupt_signal_received; ++b) {
                            p_data.dequeue(a_training_sample);
                            trainer_b.train_one_step(a_training_sample.samples, a_training_sample.labels);
                        }
                        step += b;
                        if (epoch % 50 == 0) {
                            size_t idx = rnd.get_random_32bit_number() % num_batches;                           
                            trainer_b.test_one_step(batches[idx], label_batches[idx]);
                            cout << "epoch[MAX]#: " << (epoch + 1) << "[" << num_epochs << "] step#: " <<
                                step << " learning rate: " <<
                                trainer_b.get_learning_rate() << " train loss: " <<
                                trainer_b.get_average_loss() << " test loss: " <<
                                trainer_b.get_average_test_loss() << " w/o progress: " <<
                                trainer_b.get_steps_without_progress() << endl;
                            if (trainer_b.get_learning_rate() < trainer_b.get_min_learning_rate()) break;
                        }
                    }
                    p_data.disable();
                    data_loader1.join();
                    data_loader2.join();
                    trainer_b.get_net();
                    net_b.clean();
                    serialize("slm_shakespeare_fp32_pre_v1.dat") << net_b;
                    cout << "shakespeare model saved: slm_shakespeare_fp32_pre_v1.dat" << endl;
                    cout << "shakespeare model parameters: " << count_parameters(net_b) << endl;
                    
                    deserialize("slm_shakespeare_fp32_pre_v1.dat") >> net_b;
                    visit_computational_layers(net_b, [](dropout_& l) { l = dropout_(0.0f); });
                    size_t num_correct_b = 0;
                    std::vector<unsigned long> predicted_labels_b = net_b(samples);
                    std::vector<size_t> error_indices;
                    for (size_t i = 0; i < labels.size(); ++i) {
                        if (predicted_labels_b[i] == labels[i]) num_correct_b++;
                        else error_indices.push_back(i);
                    }
                    double accuracy_b = static_cast<double>(num_correct_b) / labels.size();
                    MY_TEST_MSG(accuracy_b > 0.9, "shakespeare model (accuracy: " + to_string(accuracy_b) + ") - right: " + \
                        to_string(num_correct_b) + " - wrong: " + to_string(labels.size() - num_correct_b));
                    if (!error_indices.empty()) {
                        cout << "classification error for some sequences:" << endl;
                        for (auto idx : error_indices) cout << "  idx: " << idx << ", sample=\"" <<
                            data_b.to_unsigned_char_string(samples[idx]) << "\" => \"" <<
                            data_b.to_unsigned_char_string(labels[idx]) << "\"" << endl;
                        cout << endl;
                    }                    
                }

                // Predict the next sequence of characters
                if (fs::exists("slm_shakespeare_fp32_pre_v1.dat")) {
                    using inference_net = net_type::network_type<false>;
                    inference_net net_b;
                    deserialize("slm_shakespeare_fp32_pre_v1.dat") >> net_b;
                    std::string extracted_sequence = input_sequence.substr(input_sequence.length() - max_seq_len);
                    dlib::matrix<int, 0, 1> input_tokens;
                    input_tokens.set_size(max_seq_len);
                    for (size_t i = 0; i < max_seq_len; ++i) input_tokens(i, 0) = static_cast<int>(extracted_sequence[i]);

                    string start_seq = data_b.to_unsigned_char_string(input_tokens);
                    cout << "input sequence for text generation: <" << start_seq << "> (size: " << start_seq.length() << ")" << endl;
                    string generated_sonnet;
                    
                    for (int i = 0; i < 450; ++i) {                        
                        unsigned long next_char = net_b(input_tokens);
                        generated_sonnet += data_b.to_unsigned_char_string(next_char);

                        for (int j = 0; j < (max_seq_len - 1); ++j) input_tokens(j, 0) = input_tokens(j + 1, 0);
                        int insert_pos = std::distance(
                            input_tokens.begin(),
                            std::find_if(input_tokens.begin(), input_tokens.end(),
                                [&](const auto& element) { return element == pad_id; })
                        );
                        if (insert_pos == max_seq_len) insert_pos = max_seq_len - 1;
                        input_tokens(insert_pos, 0) = next_char;
                    }
                    input_sequence += generated_sonnet;
                    cout << "generated text:\n\n" << input_sequence << " (...)\n\n";
                    
                    {
                        // Extract latent vectors
                        /*resizable_tensor o_tensor = layer<2>(net_b).get_output();
                        const size_t num_samples = o_tensor.num_samples();
                        const size_t num_channels = o_tensor.k();
                        const size_t plane_size = o_tensor.nr() * o_tensor.nc();
                        std::vector<std::vector<float>> latent_vectors(num_samples);
                        for (size_t s = 0; s < num_samples; ++s) {
                            latent_vectors[s].resize(num_channels);
                            for (size_t k = 0; k < num_channels; ++k)
                                latent_vectors[s][k] = o_tensor.host()[tensor_index(o_tensor, s, k, 0, 0)];
                        }
                        cout << "number of latent vectors: " << latent_vectors.size() <<
                            " (input matrice: " << num_samples << "x" << num_channels << "x" <<
                            o_tensor.nr() << "x" << o_tensor.nc() << ")" << endl;
                        if (latent_vectors.size() > 0) {
                            const auto& v = latent_vectors[0];                            
                            cout << "  - values (idx:0): ";
                            if (v.size() >= (7 + 7)) {
                                for (size_t i = 0; i < 7; ++i) cout << v[i] << " ";
                                cout << " [...] ";
                                for (size_t i = v.size() - 7; i < v.size(); ++i) cout << v[i] << " ";
                            } else {
                                for (size_t i = 0; i < v.size(); ++i) cout << v[i] << " ";
                            }
                            cout << endl;
                        }*/
                    }
                }

                // Loading now the complete Shakespeare file
                string shakespeare_file = "shakespeare.txt";
                if (fs::exists(shakespeare_file)) {
                    using train_net = net_type::network_type<true>;
                    train_net net_b;
                    dnn_trainer<train_net, adam> trainer_c(net_b, adam(weight_decay, beta1, beta2), gpus);
                    trainer_c.set_learning_rate(learning_rate);
                    trainer_c.set_min_learning_rate(min_learning_rate);
                    trainer_c.set_learning_rate_shrink_factor(0.1);
                    trainer_c.set_mini_batch_size(mini_batch_size);
                    trainer_c.set_iterations_without_progress_threshold(200000);

                    // Reload previous model                    
                    if (!fs::exists("slm_shakespeare_fp32_v1.dat") && fs::exists("slm_shakespeare_fp32_pre_v1.dat")) {
                        deserialize("slm_shakespeare_fp32_pre_v1.dat") >> net_b;
                        cout << "shakespeare model loaded (source template): slm_shakespeare_fp32_pre_v1.dat" << endl;
                    }
                    else if (fs::exists("slm_shakespeare_fp32_v1.dat")) {
                        deserialize("slm_shakespeare_fp32_v1.dat") >> net_b;
                        cout << "shakespeare model loaded: slm_shakespeare_fp32_v1.dat" << endl;
                    }
                    else {
                        cout << "no previous model found, starting from scratch" << endl;
                    }

                    // Prepare data
                    documents data_c(max_seq_len, pad_id, true);
                    data_c.load_documents(shakespeare_file, false);
                    cout << "loading " << data_c.get_total_samples() << " samples from " << shakespeare_file << endl;

                    data_c.generate_samples(1, samples, labels, false);
                    size_t num_samples = (data_c.get_total_presamples() / mini_batch_size) * mini_batch_size, step = 0;
                    size_t num_batches = num_samples / mini_batch_size;
                    cout << "number of generated samples: " << num_samples << endl;
                    cout << "number of batches: " << num_batches << endl;

                    data_c.set_samples_idx(0);
                    data_c.generate_samples(num_samples, samples, labels, false);
                    std::vector<std::vector<matrix<int, 0, 1>>> batches;
                    std::vector<std::vector<unsigned long>> label_batches;
                    for (int i = 0; i < num_samples; i += mini_batch_size) {
                        std::vector<matrix<int, 0, 1>> batch_samples(samples.begin() + i, samples.begin() + i + mini_batch_size);
                        std::vector<unsigned long> batch_labels(labels.begin() + i, labels.begin() + i + mini_batch_size);
                        batches.push_back(batch_samples);
                        label_batches.push_back(batch_labels);
                    }

                    // New training loop
                    size_t num_epochs = 3000, epoch, b;
                    dlib::rand rnd(std::rand());
                    dlib::pipe<a_training> p_data(10);
                    std::thread data_loader1([&data_c, &p_data, f]() { f(data_c, p_data, true); });
                    std::thread data_loader2([&data_c, &p_data, f]() { f(data_c, p_data, true); });
                    cout << "waiting for the initial pipe loading... ";
                    while (p_data.size() < 10) std::this_thread::sleep_for(std::chrono::seconds(1));
                    cout << "done" << endl;
                    for (epoch = 0; epoch < num_epochs && !g_interrupt_signal_received; ++epoch) {
                        for (b = 0; b < num_batches && !g_interrupt_signal_received; ++b) {
                            p_data.dequeue(a_training_sample);
                            trainer_c.train_one_step(a_training_sample.samples, a_training_sample.labels);
                        }
                        step += b;
                        cout << "epoch[MAX]#: " << (epoch + 1) << "[" << num_epochs << "] step#: " <<
                            step << " learning rate: " <<
                            trainer_c.get_learning_rate() << " train loss: " <<
                            trainer_c.get_average_loss() << " test loss: " <<
                            trainer_c.get_steps_without_progress() << endl;
                        if (trainer_c.get_learning_rate() < trainer_c.get_min_learning_rate()) break;
                    }
                    p_data.disable();
                    data_loader1.join();
                    data_loader2.join();

                    // Save the new model
                    trainer_c.get_net();
                    net_b.clean();
                    serialize("slm_shakespeare_fp32_v1.dat") << net_b;
                    cout << "advanced shakespeare model saved: slm_shakespeare_fp32_v1.dat" << endl;
                    cout << "advanced shakespeare model parameters: " << count_parameters(net_b) << endl;                    

                    // Test partially the new model
                    visit_computational_layers(net_b, [](dropout_& l) { l = dropout_(0.0f); });
                    if (samples.size() > 8000) samples.resize(8000);
                    std::vector<unsigned long> predicted_labels = net_b(samples);
                    std::vector<size_t> error_indices;
                    size_t num_correct_b = 0;
                    for (size_t i = 0; i < predicted_labels.size(); ++i) {
                        if (predicted_labels[i] == labels[i]) num_correct_b++;
                        else error_indices.push_back(i);
                    }
                    double accuracy_b = static_cast<double>(num_correct_b) / predicted_labels.size();
                    MY_TEST_MSG(accuracy_b > 0.9, "advanced shakespeare model (accuracy: " + to_string(accuracy_b) + ") - right: " + \
                        to_string(num_correct_b) + " - wrong: " + to_string(predicted_labels.size() - num_correct_b));

                    // Attempting to generate a new sonnet                    
                    string sonnet_start = "Shall I compare thee to a winter's night?\nThy beauty warms the frost-bitten bough.\nIn darkness, thou art my guiding light,\nA beacon of hope 'midst winter's vow.";
                    size_t sequence_len = std::min((size_t)max_seq_len, sonnet_start.length());
                    std::string extracted_sequence = sonnet_start.substr(sonnet_start.length() - sequence_len);
                    dlib::matrix<int, 0, 1> input_tokens;
                    input_tokens.set_size(sequence_len, 1);
                    for (size_t i = 0; i < sequence_len; ++i) input_tokens(i, 0) = static_cast<unsigned char>(extracted_sequence[i]);
                    
                    cout << "\ngenerated sonnet:\n";
                    string generated_sonnet;
                    for (int i = 0; i < 600; ++i) {
                        unsigned long next_char = net_b(input_tokens);
                        generated_sonnet += data_c.to_unsigned_char_string(next_char);

                        // Stop after generating what looks like a complete sonnet
                        if (generated_sonnet.find("END") != string::npos) break;

                        for (int j = 0; j < (max_seq_len - 1); ++j) input_tokens(j, 0) = input_tokens(j + 1, 0);
                        int insert_pos = std::distance(
                            input_tokens.begin(),
                            std::find_if(input_tokens.begin(), input_tokens.end(),
                                [&](const auto& element) { return element == pad_id; })
                        );
                        if (insert_pos == max_seq_len) insert_pos = max_seq_len - 1;
                        input_tokens(insert_pos, 0) = static_cast<int>(next_char);
                    }
                    cout << sonnet_start << generated_sonnet << "\n";

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
                    };
                    int keyword_count = 0;
                    for (const auto& keyword : keywords) {
                        if (generated_sonnet.find(keyword) != string::npos) keyword_count++;
                    }
                    double relevance_score = static_cast<double>(keyword_count) / keywords.size();
                    MY_TEST_MSG(relevance_score > 0.3, "shakespeare model relevance (score: " + to_string(relevance_score) + ")");
                }
                else {
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
        using inference_network = transformer::vslm::network_type<false>;
        inference_network net;
        if (fs::exists(language_model)) deserialize(language_model) >> net;
        else {
            cerr << "language model not found! (<" << language_model << ">)" << endl;
            return 1;
        }
        visit_computational_layers(net, [](dropout_& l) { l = dropout_(0.0f); });
        cout << "number of model parameters: " << count_parameters(net) << endl << endl;
        context_window prompt(transformer::vslm::MAX_SEQ_LEN);
        string output = "";
        cout << "input prompt: (...) " << raw_data_test << " (...)" << endl;
        cout << "generated text: " << raw_data_test;

        std::vector<int> prompt_ids, endings = { eos_id, pad_id }, response_ids;
        sp.Encode(dlib::trim(raw_data_test), &prompt_ids);
        prompt.add_input(prompt_ids);
        matrix<int, 0, 1> padded_window;
        for (int i = 0; i < 150; ++i) {
            if (prompt.get_padded_window(padded_window)) {
                int next_id = static_cast<int>(net(padded_window));
                prompt.add_output(next_id);
                response_ids.push_back(next_id);
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
        std::vector<int> vocab_sizes = { 50000 };
        string corpus_files;

        for (const auto& entry : fs::recursive_directory_iterator(corpus_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                corpus_files += "\"" + entry.path().string() + "\",";
            }
        }
        corpus_files.pop_back();

        for (const auto& vocab_size : vocab_sizes) {
            string size_suffix;
            if (vocab_size == 1000) size_suffix = "1k";
            else if (vocab_size == 2000) size_suffix = "2k";
            else if (vocab_size == 4000) size_suffix = "4k";
            else if (vocab_size == 6000) size_suffix = "6k";
            else if (vocab_size == 8000) size_suffix = "8k";
            else if (vocab_size == 10000) size_suffix = "10k";
            else if (vocab_size == 25000) size_suffix = "25k";
            else if (vocab_size == 50000) size_suffix = "50k";
            string current_vocabulary_prefix = "ernie.en-fr.ung." + size_suffix;

            string train_args = "--input=" + corpus_files +
                " --model_prefix=" + current_vocabulary_prefix +
                " --bos_id=" + to_string(bos_id) + " --eos_id=" + to_string(eos_id) +
                " --unk_id=" + to_string(unk_id) + " --pad_id=" + to_string(pad_id) +
                " --user_defined_symbols=\"<rn>\"" +
                " --model_type=unigram" +
                " --character_coverage=1.0" +
                " --max_sentence_length=16768" +
                " --split_by_unicode_script=false" +
                " --input_sentence_size=10000000" +
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
        
        // Trainer setup
        const string model_sync_filename = fs::current_path().string() + "/ernie_checkpoint.dat";
        using train_network = transformer::vslm::network_type<true>;
        train_network net;
        dnn_trainer<train_network, adam> my_trainer(net, adam(weight_decay, beta1, beta2), gpus);
        my_trainer.set_learning_rate(learning_rate);
        my_trainer.set_min_learning_rate(min_learning_rate);
        my_trainer.set_iterations_without_progress_threshold(iterations_without_progress_threshold);
        my_trainer.set_mini_batch_size(mini_batch_size);
        if (use_sync_file) my_trainer.set_synchronization_file(model_sync_filename, std::chrono::minutes(5));
        if (!fs::exists(model_sync_filename) && fs::exists(language_model)) deserialize(language_model) >> net;
        std::ostringstream oss;
        oss << net << endl << my_trainer;
        cout << oss.str() << endl;
        
        // Data preparation
        documents data;
        std::vector<matrix<int, 0, 1>> samples;
        std::vector<unsigned long> labels;
        std::vector<std::vector<matrix<int, 0, 1>>> batches;
        std::vector<std::vector<unsigned long>> label_batches;
        cout << "preprocessing entries... " << endl;
        if (corpus_dir.empty()) {
            string initial_raw_data = fs::current_path().string() + "/ernie_raw_data.txt";
            write_raw_data(initial_raw_data);
            data.load_documents(initial_raw_data, false);
            fs::remove(initial_raw_data);
        } else data.load_documents(corpus_dir, false);
        data.generate_samples(1, samples, labels, false);
        size_t num_samples = (data.get_total_presamples() / mini_batch_size) * mini_batch_size, step = 0;
        size_t num_batches = num_samples / mini_batch_size;
        cout << "number of generated samples: " << num_samples << endl;
        cout << "number of batches: " << num_batches << endl;
        data.set_samples_idx(0);
        data.generate_samples(num_samples, samples, labels, false);
        for (size_t i = 0; i < num_samples; i += mini_batch_size) {
            std::vector<matrix<int, 0, 1>> batch_samples(samples.begin() + i, samples.begin() + i + mini_batch_size);
            std::vector<unsigned long> batch_labels(labels.begin() + i, labels.begin() + i + mini_batch_size);
            batches.push_back(batch_samples);
            label_batches.push_back(batch_labels);
        }

        // Use some threads to preload samples
        dlib::pipe<a_training> p_data(10);                          
        auto f = [&, mini_batch_size](documents& docs, dlib::pipe<a_training>& data, bool select_randomly) {
            a_training temp;
            while (data.is_enabled()) {
                if (docs.generate_samples(mini_batch_size, temp.samples, temp.labels, select_randomly)) data.enqueue(temp);
            }
        };       
        std::thread data_loader1([&data, &p_data, f]() { f(data, p_data, true); });
        std::thread data_loader2([&data, &p_data, f]() { f(data, p_data, true); });
        cout << "waiting for the initial pipe loading... ";
        while (p_data.size() < 10) std::this_thread::sleep_for(std::chrono::seconds(1));
        cout << "done" << endl;

        // Training loop
        size_t num_epochs = 10000, epoch, b;
        dlib::rand rnd(std::rand());
        a_training a_training_sample;
        for (epoch = 0; epoch < num_epochs && !g_interrupt_signal_received; ++epoch) {
            for (b = 0; b < num_batches && !g_interrupt_signal_received; ++b) {
                p_data.dequeue(a_training_sample);
                my_trainer.train_one_step(a_training_sample.samples, a_training_sample.labels);
            }                
            step += num_batches;
                
            size_t idx = rnd.get_random_32bit_number() % num_batches;
            my_trainer.test_one_step(batches[idx], label_batches[idx]);
            cout << "epoch[MAX]#: " << (epoch + 1) << "[" << num_epochs << "] step#: " <<
                step << " learning rate: " <<
                my_trainer.get_learning_rate() << " train loss: " <<
                my_trainer.get_average_loss() << " test loss: " <<
                my_trainer.get_average_test_loss() << " w/o progress: " <<
                my_trainer.get_steps_without_progress() << endl;
            if (my_trainer.get_learning_rate() < my_trainer.get_min_learning_rate()) break;
        }
        cout << "stopping the training process" << endl;
        p_data.disable();
        data_loader1.join();
        data_loader2.join();
        my_trainer.get_net();
        net.clean();
        cout << "model parameters: " << count_parameters(net) << endl;
        serialize(language_model) << net;
        cout << endl << "language model <" << language_model << "> saved" << endl;
        if (use_sync_file && !g_interrupt_signal_received) {
            fs::remove(model_sync_filename);
            fs::remove((string(model_sync_filename) + string("_")).c_str());
        }

        // Simple test of the model quality        
        if (fs::exists(language_model)) {
            using inference_network = transformer::vslm::network_type<false>;
            inference_network inf_net;
            deserialize(language_model) >> inf_net;
            visit_computational_layers(inf_net, [](dropout_& l) { l = dropout_(0.0f); });

            if (samples.size() > 10000) samples.resize(10000);
            std::vector<unsigned long> predicted_labels = inf_net(samples);
            std::vector<size_t> error_indices;
            size_t num_correct = 0;
            for (size_t i = 0; i < predicted_labels.size(); ++i) {
                if (predicted_labels[i] == labels[i]) num_correct++;
                else error_indices.push_back(i);
            }
            double accuracy_b = static_cast<double>(num_correct) / predicted_labels.size();
            MY_TEST_MSG(accuracy_b > 0.9, "model (accuracy: " + to_string(accuracy_b) + ") - right: " + \
                to_string(num_correct) + " - wrong: " + to_string(predicted_labels.size() - num_correct));
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

        using inference_network = transformer::vslm::network_type<false>;
        inference_network net;
        if (fs::exists(language_model)) deserialize(language_model) >> net;
        else {
            cerr << "language model not found! (<" << language_model << ">)" << endl;
            return 1;
        }
        float drop_rate = static_cast<float>((1.0 - temperature) / 2.0f);
        visit_computational_layers(net, [&](dropout_& l) { l = dropout_(drop_rate); });
        cout << "number of model parameters: " << count_parameters(net) << endl;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(transformer::vslm::MAX_SEQ_LEN /2, transformer::vslm::MAX_SEQ_LEN *6);
        context_window prompt(transformer::vslm::MAX_SEQ_LEN);
        std::vector<int> prompt_ids, endings = { eos_id, pad_id }, response_ids;

        cout << ">>> press [CTRL+C] to stop the dialog with ERNIE <<<" << endl << endl;
        string input, output;
        cout << "[YOU] ";
        std::getline(std::cin, input);
        do {            
            if (!g_interrupt_signal_received && !input.empty()) {
                // Encode and extract IDs
                sp.Encode(dlib::trim(input), &prompt_ids);
                prompt.reset();
                prompt.add_input(prompt_ids);
                size_t total_steps = std::min(static_cast<int>(transformer::vslm::MAX_SEQ_LEN), dis(gen));
                // Generate response
                response_ids.clear();
                cout << "[ERNIE] ";
                matrix<int, 0, 1> padded_window;
                int cur_top_k = (top_k >= 1) ? top_k : 1, predicted_label;
                for (int i = 0; i < total_steps; ++i) {                    
                    if (prompt.get_padded_window(padded_window)) {
                        predicted_label = static_cast<int>(net(padded_window));

                        /*matrix<float, llm::sequence_size, llm::vocab_size> output = mat(generator(padded_window));
                        matrix<float, 1, llm::vocab_size> logits = sum_rows(output);
                        if (cur_top_k <= 1) {
                            predicted_label = index_of_max(logits);
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
                        }*/
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