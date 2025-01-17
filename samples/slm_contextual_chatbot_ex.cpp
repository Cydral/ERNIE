#include <iostream>
#include <iomanip>
#include <string>
#include <locale>
#include <codecvt>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <fstream>
#include <cmath>
#include <random>
#include <regex>
#ifdef _WIN32
#include <csignal>
#endif
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/misc_api.h>
#include <dlib/base64.h>
#include <dlib/dir_nav.h>

// Include Transformer definitions
#include "slm_defs.h"

// This header "slm_data.h" is assumed to contain at least:
//   const std::string shakespeare_text;
#include "slm_data.h"

const size_t VOCAB_SIZE = 1000;
const size_t NUM_SPECIAL_TOKENS = 3;

// Initialize random number generation
std::random_device rd;
std::mt19937 gen(rd());

// ----------------------------------------------------------------------------------------
std::atomic<bool> g_interrupt_signal_received{ false };
#ifdef _WIN32
void signalHandler(int signal) {
    if (signal == SIGINT) {
        g_interrupt_signal_received.store(true, std::memory_order_relaxed);
        std::cout << "\ninterrupt detected (CTRL+C), cleaning up and closing the program" << std::endl;
    }
}
#endif

// ----------------------------------------------------------------------------------------
// BPE Tokenizer class
class bpe_tokenizer {
public:
    bpe_tokenizer() : vocab_size(BASE_VOCAB_SIZE) {
        // Initialize the base vocabulary with single bytes
        for (int i = 0; i < BASE_VOCAB_SIZE; ++i) {
            vocab[i] = std::vector<uint8_t>{ static_cast<uint8_t>(i) };
        }
        // Initialize special tokens with sequential IDs
        special_tokens = {
            {"<|endoftext|>", BASE_VOCAB_SIZE},
            {"<|unk|>", (BASE_VOCAB_SIZE + 1)},
            {"<|pad|>", (BASE_VOCAB_SIZE + 2)}
        };
    }

    // Train the tokenizer on the given text
    void train(const std::string& text, int vocab_size, bool verbose = false) {
        assert(vocab_size >= BASE_VOCAB_SIZE);
        this->vocab_size = vocab_size;
        int num_merges = vocab_size - BASE_VOCAB_SIZE;

        // Convert text to byte IDs
        std::vector<int> ids;
        for (char c : text) {
            ids.push_back(static_cast<uint8_t>(c));
        }

        // Perform BPE merges
        for (int i = 0; i < num_merges; ++i) {
            auto stats = get_stats(ids);
            if (stats.empty()) break;

            // Find the most frequent pair that does not exceed MAX_TOKEN_LENGTH
            auto pair = std::max_element(stats.begin(), stats.end(),
                [this](const std::pair<std::pair<int, int>, int>& a, const std::pair<std::pair<int, int>, int>& b) {
                    // Check if the resulting token would exceed MAX_TOKEN_LENGTH
                    size_t a_length = vocab[a.first.first].size() + vocab[a.first.second].size();
                    size_t b_length = vocab[b.first.first].size() + vocab[b.first.second].size();
                    if (a_length > MAX_TOKEN_LENGTH) return true;  // Skip pair a
                    if (b_length > MAX_TOKEN_LENGTH) return false; // Skip pair b
                    return a.second < b.second;
                })->first;

            // Check if the resulting token would exceed MAX_TOKEN_LENGTH
            size_t new_token_length = vocab[pair.first].size() + vocab[pair.second].size();
            if (new_token_length > MAX_TOKEN_LENGTH) {
                if (verbose) {
                    std::cout << "\r"
                        << std::setw(100) << std::flush
                        << "\rskipping merge " << std::to_string(i + 1) << "/" << std::to_string(num_merges) << ": ("
                        << std::to_string(pair.first) << "," << std::to_string(pair.second) << ") -> new token length "
                        << std::to_string(new_token_length) << " exceeds limit of " << std::to_string(MAX_TOKEN_LENGTH)
                        << std::flush;
                }
                continue; // Skip this merge
            }

            int idx = (BASE_VOCAB_SIZE + (int)special_tokens.size()) + i;
            ids = merge(ids, pair, idx);
            merges[pair] = idx;
            vocab[idx].insert(vocab[idx].end(), vocab[pair.first].begin(), vocab[pair.first].end());
            vocab[idx].insert(vocab[idx].end(), vocab[pair.second].begin(), vocab[pair.second].end());

            if (verbose) {
                std::cout << "\r"
                    << std::setw(100) << std::flush
                    << "\rmerge " << std::to_string(i + 1) << "/" << std::to_string(num_merges) << ": ("
                    << std::to_string(pair.first) << "," << std::to_string(pair.second) << ") -> " << std::to_string(idx)
                    << " (" << bytes_to_string(vocab[idx]) << ") had "
                    << std::to_string(stats[pair]) << " occurrences"
                    << std::flush;
            }
        }
        std::cout << "\ntraining done\n";
    }

    // Encode the given text into subword tokens
    std::vector<int> encode(const std::string& text) {
        std::vector<int> ids;
        for (char c : text) {
            ids.push_back(static_cast<uint8_t>(c));
        }

        while (ids.size() >= 2) {
            auto stats = get_stats(ids);
            auto pair = std::min_element(stats.begin(), stats.end(),
                [this](const std::pair<std::pair<int, int>, int>& a, const std::pair<std::pair<int, int>, int>& b) {
                    return merges.count(a.first) ? (merges.at(a.first) < (merges.count(b.first) ? merges.at(b.first) : INT_MAX)) : false;
                })->first;

            if (!merges.count(pair)) break;
            int idx = merges[pair];
            ids = merge(ids, pair, idx);
        }

        return ids;
    }

    // Decode a single token ID back into text
    std::string decode(int id) {
        for (const auto& token : special_tokens) {
            if (token.second == id) {
                return token.first;
            }
        }
        auto& token = vocab.at(id);
        return std::string(token.begin(), token.end());
    }

    // Decode a sequence of token IDs back into text
    std::string decode(const std::vector<int>& ids) {
        std::vector<uint8_t> bytes;
        for (int id : ids) {
            bool is_special_token = false;
            for (const auto& token : special_tokens) {
                if (token.second == id) {
                    bytes.insert(bytes.end(), token.first.begin(), token.first.end());
                    is_special_token = true;
                    break;
                }
            }
            if (!is_special_token) {
                auto& token = vocab.at(id);
                bytes.insert(bytes.end(), token.begin(), token.end());
            }
        }
        return std::string(bytes.begin(), bytes.end());
    }

    // Save the tokenizer model and vocabulary to files
    void save(const std::string& file_prefix) {
        std::ofstream model_file(file_prefix + ".model");
        model_file << "bpe-tokenizer v1\n";

        for (int idx = BASE_VOCAB_SIZE; idx < vocab_size; ++idx) {
            for (const auto& merge_pair : merges) {
                if (merge_pair.second == idx) {
                    model_file << merge_pair.first.first << " " << merge_pair.first.second << "\n";
                    break;
                }
            }
        }

        std::ofstream vocab_file(file_prefix + ".vocab");
        for (const auto& v : vocab) {
            vocab_file << "[" << bytes_to_string(v.second) << "] " << v.first << "\n";
        }
    }

    // Load the tokenizer model and vocabulary from files
    bool load(const std::string& model_file) {
        std::ifstream file(model_file + ".model");
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open model file: " << model_file + ".model" << "\n";
            return false;
        }

        std::string line;
        std::getline(file, line); // Version

        merges.clear();
        vocab.clear();
        for (int i = 0; i < BASE_VOCAB_SIZE; ++i) {
            vocab[i] = std::vector<uint8_t>{ static_cast<uint8_t>(i) };
        }

        int idx = BASE_VOCAB_SIZE + (int)special_tokens.size();
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            int a, b;
            iss >> a >> b;
            merges[{a, b}] = idx;

            vocab[idx].insert(vocab[idx].end(), vocab[a].begin(), vocab[a].end());
            vocab[idx].insert(vocab[idx].end(), vocab[b].begin(), vocab[b].end());
            idx++;
        }

        std::ifstream vocab_file(model_file + ".vocab");
        if (!vocab_file.is_open()) {
            std::cerr << "Error: Unable to open vocab file: " << model_file + ".vocab" << "\n";
            return false;
        }
        while (std::getline(vocab_file, line)) {
            size_t start = line.find('[');
            size_t end = line.find(']');
            if (start != std::string::npos && end != std::string::npos) {
                std::string token_str = line.substr(start + 1, end - start - 1);
                int id = std::stoi(line.substr(end + 2));
                vocab[id] = string_to_bytes(token_str);
            }
        }
        return true;
    }

    // Get the ID of a special token
    int get_special_token_id(const std::string& token) const {
        auto it = special_tokens.find(token);
        if (it != special_tokens.end()) {
            return it->second;
        }
        throw std::runtime_error("Special token not found: " + token);
    }

    // Get the total vocabulary size
    size_t get_vocab_size(void) const {
        return (vocab.size() + special_tokens.size());
    }

private:
    std::map<std::string, int> special_tokens;
    std::map<std::pair<int, int>, int> merges;
    std::map<int, std::vector<uint8_t>> vocab;
    int vocab_size;

    static const size_t MAX_TOKEN_LENGTH = 8;
    static const int BASE_VOCAB_SIZE = 256;

    // Get frequency statistics of adjacent token pairs
    std::map<std::pair<int, int>, int> get_stats(const std::vector<int>& ids) {
        std::map<std::pair<int, int>, int> stats;
        for (size_t i = 0; i < ids.size() - 1; ++i) {
            stats[{ids[i], ids[i + 1]}]++;
        }
        return stats;
    }

    // Merge the most frequent pair in the token sequence
    std::vector<int> merge(const std::vector<int>& ids, std::pair<int, int> pair, int idx) {
        std::vector<int> new_ids;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i < ids.size() - 1 && ids[i] == pair.first && ids[i + 1] == pair.second) {
                new_ids.push_back(idx);
                i++;
            }
            else new_ids.push_back(ids[i]);
        }
        return new_ids;
    }

    // Convert a sequence of bytes to a readable string
    std::string bytes_to_string(const std::vector<uint8_t>& bytes) {
        std::string result;
        for (uint8_t byte : bytes) {
            if (byte >= 32 && byte <= 126) {
                result += static_cast<char>(byte);
            }
            else {
                char buf[5];
                snprintf(buf, sizeof(buf), "\\x%02x", byte);
                result += buf;
            }
        }
        return result;
    }

    // Convert a string representation of bytes back to bytes
    std::vector<uint8_t> string_to_bytes(const std::string& str) {
        std::vector<uint8_t> bytes;
        for (size_t i = 0; i < str.length(); ++i) {
            if (str[i] == '\\' && i + 3 < str.length() && str[i + 1] == 'x') {
                char hex[3] = { str[i + 2], str[i + 3], '\0' };
                uint8_t byte = static_cast<uint8_t>(std::stoul(hex, nullptr, 16));
                bytes.push_back(byte);
                i += 3;
            }
            else {
                bytes.push_back(static_cast<uint8_t>(str[i]));
            }
        }
        return bytes;
    }
};

// ----------------------------------------------------------------------------------------

dlib::matrix<int, 0, 1> create_sample_with_unknown_tokens(const dlib::matrix<int, 0, 1>& o_sample, int num_unknown_tokens, int unknown_token_id) {
    dlib::matrix<int, 0, 1> new_sample = o_sample;
    std::uniform_int_distribution<> dis(0, o_sample.size() - 1);

    for (int i = 0; i < num_unknown_tokens; ++i)
        new_sample(dis(gen)) = unknown_token_id;

    return new_sample;
}

// ----------------------------------------------------------------------------------------

std::pair<dlib::matrix<int, 0, 1>, int> create_reduced_sample(const dlib::matrix<int, 0, 1>& o_sample, int padding_token_id, double reduction_ratio) {
    int new_length = o_sample.size() * (1 - reduction_ratio);
    dlib::matrix<int, 0, 1> new_sample = o_sample;

    for (int i = new_length; i < o_sample.size(); ++i)
        new_sample(i) = padding_token_id;

    return { new_sample, o_sample(new_length) };
}

// ----------------------------------------------------------------------------------------

std::string load_data_from_file_or_directory(const std::string& path, size_t max_size = 0.05 * 1024 * 1024) {
    std::string data;
    size_t total_size = 0;
    bool max_size_reached = false;
    const size_t buffer_size = 4 * 1024;

    auto process_file = [&](const std::string& file_path) {
        std::ifstream input(file_path, std::ios::binary);
        if (input.is_open()) {
            std::cout << "Loading file: " << file_path << std::endl;

            std::vector<char> buffer(buffer_size);
            bool first_chunk = true;

            while (input.read(buffer.data(), buffer_size) || input.gcount() > 0) {
                size_t bytes_read = input.gcount();

                if (!max_size_reached) {
                    size_t remaining_space = max_size - total_size;
                    size_t bytes_to_add = std::min(remaining_space, bytes_read);

                    if (bytes_to_add > 0) {
                        if (!first_chunk && !data.empty()) data += "\n\n";
                        data.append(buffer.data(), bytes_to_add);
                        total_size += bytes_to_add;
                        first_chunk = false;
                    }

                    if (total_size >= max_size) {
                        max_size_reached = true;
                        std::cout << "Max size limit reached. Further content will be ignored." << std::endl;
                        break;
                    }
                }
                else {
                    break;  // No need to continue reading if max size is reached
                }
            }
        }
        };

    try {
        dlib::directory dir(path);
        std::vector<dlib::file> files = dir.get_files();
        for (const auto& file : files) {
            if (max_size_reached) break;
            process_file(file.full_name());
        }
    }
    catch (const dlib::directory::dir_not_found) {
        process_file(path);
    }

    std::cout << "Total data size: " << total_size << " bytes" << std::endl;
    return data;
}

// ----------------------------------------------------------------------------------------

// Function to shuffle samples and labels in sync
void shuffle_samples_and_labels(std::vector<dlib::matrix<int, 0, 1>>& samples, std::vector<unsigned long>& labels) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::vector<size_t> indices(samples.size());
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ..., N-1
    std::shuffle(indices.begin(), indices.end(), generator);

    // Apply the shuffle
    for (size_t i = 0; i < indices.size(); ++i) {
        while (i != indices[i]) {
            size_t j = indices[i];
            std::swap(samples[i], samples[j]);
            std::swap(labels[i], labels[j]);
            std::swap(indices[i], indices[j]);
        }
    }
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) {
    try {
        dlib::command_line_parser parser;
        parser.add_option("train", "Train a small transformer on the built-in Shakespeare text");
        parser.add_option("chatbot", "Engage in an interactive chatbot session using a trained model");
        parser.add_option("test-tokenizer", "Test the BPE tokenizer");
        parser.add_option("data", "Specify a file or directory containing the training data", 1);
        parser.add_option("learning-rate", "Set the learning rate for training (default: 1e-4)", 1);
        parser.add_option("batch-size", "Set the mini-batch size for training (default: 64)", 1);
        parser.add_option("max-epochs", "Set the maximum number of training epochs (default: 100)", 1);
        parser.add_option("iterations-threshold", "Set the iterations without progress threshold (default: 15000)", 1);
        parser.add_option("min-learning-rate", "Set the minimum learning rate (default: 1e-6)", 1);
        parser.add_option("shuffle", "Shuffle training sequences and labels before training (default: false)");
        parser.add_option("temperature", "Set the temperature for text generation (default: 1.0)", 1);
        parser.add_option("top-k", "Set the number of top tokens to considere for response initialization (default: 1)", 1);
        parser.parse(argc, argv);

        if (parser.number_of_arguments() == 0 && !parser.option("train")
            && !parser.option("chatbot")
            && !parser.option("test-tokenizer")) {
            std::cout << "Usage:\n"
                << "  --train: Train a Dlib transformer model on specific text\n"
                << "  --chatbot: Engage in an interactive chatbot session using a trained model\n"
                << "  --test-tokenizer: Test the BPE tokenizer\n"
                << "  --data <path>: Specify a file or directory containing the training data\n"
                << "  --learning-rate <value>: Set the learning rate for training (default: 1e-4)\n"
                << "  --batch-size <value>: Set the mini-batch size for training (default: 64)\n"
                << "  --max-epochs <value>: Set the maximum number of training epochs (default: 100)\n"
                << "  --iterations-threshold <value>: Set the iterations without progress threshold (default: 15000)\n"
                << "  --min-learning-rate <value>: Set the minimum learning rate (default: 1e-6)\n"
                << "  --shuffle: Shuffle training sequences and labels before training (default: false)\n"
                << "  --temperature: Set the temperature for text generation (default: 1.0)\n"
                << "  --top-k: Set the number of top tokens to considere for response initialization (default: 1)\n";
            return 0;
        }

        // Test the tokenizer
        if (parser.option("test-tokenizer")) {
            bpe_tokenizer c_tokenizer;
            c_tokenizer.load("dlib_t1k_base");

            // Test the tokenizer
            std::string input = "The quick brown fox jumps over the lazy dog, and the dog barks loudly!";
            auto encoded = c_tokenizer.encode(input);
            std::cout << "Encoded tokens: ";
            for (int id : encoded) std::cout << id << " ";
            std::cout << std::endl;
            encoded[3] = c_tokenizer.get_special_token_id("<|unk|>");
            encoded.push_back(c_tokenizer.get_special_token_id("<|endoftext|>"));
            encoded.push_back(c_tokenizer.get_special_token_id("<|pad|>"));

            std::string decoded = c_tokenizer.decode(encoded);
            std::cout << "Decoded string: " << decoded << std::endl;

            return 0;
        }

        // Default values
        double learning_rate = 1e-4;
        long batch_size = 64;
        int max_epochs = 100;
        int iterations_threshold = 15000;
        double min_learning_rate = 1e-6;
        double temperature = 1.0;
        int top_k = 1;

        // Override defaults if options are provided
        if (parser.option("learning-rate"))
            learning_rate = std::stod(parser.option("learning-rate").argument());
        if (parser.option("batch-size"))
            batch_size = std::stol(parser.option("batch-size").argument());
        if (parser.option("max-epochs"))
            max_epochs = std::stoi(parser.option("max-epochs").argument());
        if (parser.option("iterations-threshold"))
            iterations_threshold = std::stoi(parser.option("iterations-threshold").argument());
        if (parser.option("min-learning-rate"))
            min_learning_rate = std::stod(parser.option("min-learning-rate").argument());
        if (parser.option("temperature"))
            temperature = std::stod(parser.option("temperature").argument());
        if (parser.option("top-k"))
            top_k = std::stoi(parser.option("top-k").argument());

        // Minimal configiguration for our Transformer mmodel
        const long vocab_size = (VOCAB_SIZE + NUM_SPECIAL_TOKENS);
        const long num_layers = 10;
        const long num_heads = 8;
        const long embedding_dim = 72;
        const long max_seq_len = 80;
        const bool use_squeezing = false;

        using my_transformer_cfg = transformer::transformer_config<
            vocab_size,
            num_layers,
            num_heads,
            embedding_dim,
            max_seq_len,
            use_squeezing,
            dlib::gelu,
            dlib::dropout_10
        >;

        // For GPU usage (if any), set gpus = {0} for a single GPU, etc.
        std::vector<int> gpus{ 0 };

        // The model file to store or load (and the checkpoint)
        const std::string model_file = "lm_chatbot_en_35M_fp32_model.dat";
        const std::string checkpoint_file = "checkpoint.dat";

        // Load the tokenizer
        bpe_tokenizer tokenizer;
        if (!tokenizer.load("dlib_t1k_base")) {
            std::cerr << "Failed to load vocabulary file (dlib_t1k_base.[model|vocab])!" << std::endl;
            return 1;
        } else {
            std::cout << "Vocab size: " << std::to_string(tokenizer.get_vocab_size()) << " - Model config: "
                << std::to_string(VOCAB_SIZE + NUM_SPECIAL_TOKENS) << std::endl;
        }

        // ----------------------------------------------------------------------------------------
        // Train mode
        // ----------------------------------------------------------------------------------------
        if (parser.option("train")) {
            std::cout << "=== TRAIN MODE ===\n";

            // Set the signal handler for SIGINT
#ifdef _WIN32
            signal(SIGINT, signalHandler);
#endif

            // Load data from the specified file or directory
            std::string training_data;
            if (parser.option("data"))
                training_data = load_data_from_file_or_directory(parser.option("data").argument(), 50 * 1024 * 1024);
            else                
                training_data = shakespeare_text_parts[0]; // Fallback to the default data from slm_data.h

            // 1) Tokenize the Shakespeare text
            std::cout << "Encoding sequences in progress...";
            std::vector<int> full_tokens = tokenizer.encode(training_data);
            std::cout << " done\n";
            if (full_tokens.empty()) return 0;

            // Calculate the maximum number of sequences
            size_t max_sequences = (full_tokens.size() > (size_t)max_seq_len + 1)
                ? (full_tokens.size() - ((size_t)max_seq_len + 1))
                : 0;

            // Display the size of the training text and the number of sequences
            std::cout << "Training model on " << full_tokens.size() << " tokens\n";
            std::cout << "Maximum number of sequences: " << max_sequences << "\n";

            // Check if the text is too short
            if (max_sequences == 0) return 0;

            std::vector<dlib::matrix<int, 0, 1>> samples;
            std::vector<unsigned long> labels;

            // Let's create a training set of about (N) samples from the data
            for (size_t start = 0; start < max_sequences; ++start) {
                dlib::matrix<int, 0, 1> seq(max_seq_len, 1);
                for (long t = 0; t < max_seq_len; ++t) seq(t, 0) = full_tokens[start + t];
                samples.push_back(seq);
                labels.push_back(full_tokens[start + max_seq_len]);
            }

            // Calculate the number of additional samples (25% of the original dataset)
            size_t additional_samples = max_sequences * 0.25;

            // Define distributions for random sampling
            std::uniform_int_distribution<> dis_sample(0, samples.size() - 1);  // For selecting original samples
            std::uniform_int_distribution<> dis_unknown(1, 5);  // For selecting number of unknown tokens (1 to 5)
            std::uniform_real_distribution<> dis_reduction(0.15, 0.85);  // For selecting reduction ratio (15% to 85%)

            // Generate additional samples
            for (size_t i = 0; i < additional_samples; ++i) {
                // Randomly select an original sample
                int o_index = dis_sample(gen);
                if (i % 2 == 0) {
                    // Create a sample with unknown tokens (50% of additional samples)
                    dlib::matrix<int, 0, 1> new_sample = create_sample_with_unknown_tokens(
                        samples[o_index],
                        dis_unknown(gen),
                        tokenizer.get_special_token_id("<|unk|>")
                    );
                    samples.push_back(new_sample);
                    labels.push_back(labels[o_index]);
                } else {
                    // Create a reduced sample (50% of additional samples)
                    double reduction_ratio = dis_reduction(gen);
                    std::pair<dlib::matrix<int, 0, 1>, int> result = create_reduced_sample(
                        samples[o_index],
                        tokenizer.get_special_token_id("<|pad|>"),
                        reduction_ratio
                    );
                    samples.push_back(result.first);
                    labels.push_back(result.second);
                }
            }

            // Update the total number of sequences after augmentation
            max_sequences = samples.size();
            std::cout << "Number of sequences after augmentation: " << max_sequences << "\n";

            // 2) Construct the network in training mode
            using net_type = my_transformer_cfg::network_type<true>;
            net_type net;
            if (!dlib::file_exists(checkpoint_file) && dlib::file_exists(model_file)) {
                std::cout << "Loading existing model...";
                dlib::deserialize(model_file) >> net;
                std::cout << " done\n";
            }

            // 3) Create dnn_trainer
            dlib::dnn_trainer<net_type, dlib::adam> trainer(net, dlib::adam(learning_rate, 0.9, 0.999), gpus);
            trainer.set_learning_rate(learning_rate);
            trainer.set_min_learning_rate(min_learning_rate);
            trainer.set_iterations_without_progress_threshold(iterations_threshold);
            trainer.set_synchronization_file(checkpoint_file, std::chrono::minutes(5));
            trainer.be_quiet();

            // 4) Main train loop            
            for (size_t epoch = 0; epoch < max_epochs
                && !g_interrupt_signal_received.load(std::memory_order_relaxed); ++epoch) {
                // Shuffle samples and labels if the --shuffle option is enabled
                if (parser.option("shuffle")) shuffle_samples_and_labels(samples, labels);

                // Calculate the number of complete batches
                size_t num_complete_batches = samples.size() / batch_size;                

                // Iterate on complete batches only
                auto last_print_time = std::chrono::steady_clock::now();
                for (size_t i = 0; i < (num_complete_batches * batch_size) && !g_interrupt_signal_received.load(std::memory_order_relaxed); i += batch_size) {
                    std::vector<dlib::matrix<int, 0, 1>> batch_samples(samples.begin() + i, samples.begin() + i + batch_size);
                    std::vector<unsigned long> batch_labels(labels.begin() + i, labels.begin() + i + batch_size);
                    trainer.train_one_step(batch_samples, batch_labels);

                    // Display the progress
                    auto current_time = std::chrono::steady_clock::now();
                    if (std::chrono::duration_cast<std::chrono::seconds>(current_time - last_print_time).count() >= 15) {
                        std::uniform_int_distribution<> dis(0, samples.size() - 1);
                        size_t random_index = dis(gen);
                        std::string decoded_sequence;
                        for (long t = 0; t < (max_seq_len < 20L ? max_seq_len : 20L); ++t)
                            decoded_sequence += tokenizer.decode(samples[random_index](t, 0));
                        std::string decoded_label = tokenizer.decode(labels[random_index]);
                        std::cout << "(sample#" << random_index << ") \"" << decoded_sequence << "(...)\" => \"" << decoded_label << "\"\n";
                        std::cout << "epoch: " << (epoch + 1) << "/" << max_epochs
                            << "\tstep: " << trainer.get_train_one_step_calls()
                            << "\tlearning rate: " << trainer.get_learning_rate()
                            << "\taverage loss: " << trainer.get_average_loss()
                            << "\tsteps without progress: " << trainer.get_steps_without_progress()
                            << std::flush << std::endl;
                        last_print_time = current_time;                        
                    }
                }
                if (trainer.get_learning_rate() < trainer.get_min_learning_rate()) break;
            }

            // 5) Evaluate quickly on the training set
            if (samples.size() > 20000) samples.resize(20000);
            auto predicted = net(samples);
            size_t correct = 0;
            for (size_t i = 0; i < samples.size(); ++i)
                if (predicted[i] == labels[i]) correct++;
            double accuracy = (double)correct / samples.size();
            std::cout << "Training accuracy (on this sample set): " << accuracy << 
                " (correct: " << correct << " - error: " << (samples.size() - correct) << ")\n";

            // 6) Save the model
            net.clean();
            dlib::serialize(model_file) << net;
            std::cout << "Model saved to " << model_file << "\n";

            // 7) Remove checkpoints
            std::remove(checkpoint_file.c_str());
            std::remove((checkpoint_file + "_").c_str());
        }

        // ----------------------------------------------------------------------------------------
        // Chatbot mode
        // ----------------------------------------------------------------------------------------
        if (parser.option("chatbot")) {
            std::cout << "=== CHATBOT MODE ===\n";

            // 1) Load the trained model
            using net_infer = my_transformer_cfg::network_type<false>;
            net_infer net;
            dlib::softmax<dlib::multiply<net_infer::subnet_type>> chatbot(dlib::multiply_(1.0 / temperature));

            if (dlib::file_exists(model_file)) {
                dlib::deserialize(model_file) >> net;
                std::cout << "Loaded model from " << model_file << "\n";
            } else {
                std::cerr << "Error: model file not found. Please run --train first.\n";
                return 0;
            }
            std::cout << my_transformer_cfg::model_info::describe() << std::endl;
            std::cout << "Model parameters: " << count_parameters(net) << std::endl << std::endl;

            // 2) Initialize the conversation history and generator
            std::vector<int> conversation_tokens;
            chatbot.subnet().subnet() = net.subnet();

            // 3) Conversation loop
            const float top_k_threshold = 0.05f; // 5% threshold to consider a token as a candidate
            std::string user_input;
            while (true) {
                // Prompt the user for input
                std::cout << "You: ";
                std::getline(std::cin, user_input);

                // Exit if the user types "bye"
                if (user_input == "bye") {
                    std::cout << "Chatbot: Goodbye!\n";
                    break;
                }

                // Append the user's input to the conversation history
                std::vector<int> user_tokens = tokenizer.encode(user_input + "\n");
                conversation_tokens.insert(conversation_tokens.end(), user_tokens.begin(), user_tokens.end());

                // Truncate the conversation history to fit within max_seq_len
                if (conversation_tokens.size() > (size_t)max_seq_len)
                    conversation_tokens.erase(conversation_tokens.begin(), conversation_tokens.end() - max_seq_len);

                // Create a dlib matrix for the model input
                dlib::matrix<int, 0, 1> input_seq(max_seq_len, 1);
                for (long i = 0; i < max_seq_len; ++i) {
                    if ((size_t)i < conversation_tokens.size())
                        input_seq(i, 0) = conversation_tokens[i];
                    else
                        input_seq(i, 0) = tokenizer.get_special_token_id("<|pad|>"); // Use the padding token ID
                }

                // Generate a response
                std::string generated_text;
                bool stop_generation = false;
                const size_t min_response_length = user_tokens.size(); // Minimum response length = user input length
                const size_t max_response_length = max_seq_len / 2;    // Maximum response length = half of context window
                size_t response_length = 0;

                while (!stop_generation) {
                    auto logits = dlib::mat(chatbot(input_seq)); // Single inference
                    
                    // Apply the top-k mechanism for the first token
                    unsigned long next_token;
                    if (response_length == 0 && top_k > 1) {
                        // Find the top-k tokens
                        std::vector<std::pair<float, unsigned long>> token_probs;
                        for (long i = 0; i < logits.nc(); ++i)
                            token_probs.emplace_back(logits(0, i), i);                        
                        std::partial_sort(token_probs.begin(), token_probs.begin() + top_k, token_probs.end(),
                            [](const auto& a, const auto& b) { return a.first > b.first; });

                        // Find the number of tokens that exceed the threshold
                        size_t valid_tokens = 0;
                        for (size_t i = 0; i < top_k; ++i) {
                            if (token_probs[i].first >= top_k_threshold) valid_tokens++;                            
                            else break; // Stop when we find a token below the threshold
                        }

                        // If no tokens exceed the threshold, just use the top token
                        if (valid_tokens == 0) next_token = token_probs[0].second;
                        else {
                            // Randomly select from the valid tokens
                            std::uniform_int_distribution<> dist(0, valid_tokens - 1);
                            next_token = token_probs[dist(gen)].second;
                        }
                    } else next_token = dlib::index_of_max(logits);

                    // Decode the token to a string
                    std::string next_word = tokenizer.decode(next_token);
                    generated_text += next_word;
                    response_length++;

                    // Shift the input sequence to the left
                    for (long i = 0; i < max_seq_len - 1; ++i)
                        input_seq(i, 0) = input_seq(i + 1, 0);
                    input_seq(max_seq_len - 1, 0) = (int)next_token;

                    // Check for stopping conditions
                    if (response_length >= min_response_length) {
                        // Stop if a sentence-ending punctuation mark is encountered
                        if (next_word.find_first_of(".!?") != std::string::npos)
                            stop_generation = true;
                    }

                    // Stop if the maximum response length is reached
                    if (response_length >= max_response_length)
                        stop_generation = true;
                }

                // Display the chatbot's response
                std::cout << "Chatbot: " << generated_text << std::flush << std::endl;

                // Append the chatbot's response to the conversation history
                std::vector<int> response_tokens = tokenizer.encode(generated_text + "\n");
                conversation_tokens.insert(conversation_tokens.end(), response_tokens.begin(), response_tokens.end());

                // Truncate the conversation history again to fit within max_seq_len
                if (conversation_tokens.size() > (size_t)max_seq_len)
                    conversation_tokens.erase(conversation_tokens.begin(), conversation_tokens.end() - max_seq_len);
            }
        }

        return 0;
    }
    catch (std::exception& e) {
        std::cerr << "Exception thrown: " << e.what() << std::endl;
        return 1;
    }
}