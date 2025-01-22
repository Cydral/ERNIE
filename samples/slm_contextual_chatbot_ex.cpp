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

// Include at least data for the training
#include "slm_contextual_chatbot_data.h"

const size_t VOCAB_SIZE = 3000;
const size_t NUM_SPECIAL_TOKENS = 6;

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
            {"<|startoftext|>", BASE_VOCAB_SIZE},
            {"<|endoftext|>", BASE_VOCAB_SIZE + 1},
            {"<|question|>", BASE_VOCAB_SIZE + 2},
            {"<|response|>", BASE_VOCAB_SIZE + 3},
            {"<|unk|>", (BASE_VOCAB_SIZE + 4)},
            {"<|pad|>", (BASE_VOCAB_SIZE + 5)}
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
            auto pair = get_most_frequent_pair(stats);

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
        // Convert text to byte IDs
        std::vector<int> ids;
        ids.reserve(text.size()); // Reserve space to avoid reallocations
        for (char c : text) {
            ids.push_back(static_cast<uint8_t>(c));
        }

        // Precompute valid pairs and their merge order
        auto stats = get_stats(ids); // Compute initial statistics
        std::priority_queue<std::pair<int, std::pair<int, int>>> pq; // Min-heap based on merge order

        // Initialize the priority queue with valid pairs
        for (const auto& stat : stats) {
            const std::pair<int, int>& pair = stat.first;
            if (merges.count(pair)) {
                pq.push({ merges.at(pair), pair }); // Use merge order as the key
            }
        }

        // Merge pairs in order of their merge priority
        while (!pq.empty()) {
            const auto& top_element = pq.top(); // Get the pair with the smallest merge order
            int merge_order = top_element.first;
            const std::pair<int, int>& pair = top_element.second;
            pq.pop();

            // Check if the pair still exists in the current ids sequence
            bool pair_found = false;
            for (size_t i = 0; i < ids.size() - 1; ++i) {
                if (ids[i] == pair.first && ids[i + 1] == pair.second) {
                    pair_found = true;
                    break;
                }
            }
            if (!pair_found) continue; // Skip if the pair no longer exists

            // Merge the pair
            int idx = merges.at(pair);
            ids = merge(ids, pair, idx); // Use an optimized merge function

            // Update statistics and priority queue with new pairs formed after merging
            stats = get_stats(ids);
            for (const auto& stat : stats) {
                const std::pair<int, int>& new_pair = stat.first;
                if (merges.count(new_pair)) {
                    pq.push({ merges.at(new_pair), new_pair });
                }
            }
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

        for (int idx = BASE_VOCAB_SIZE + (int)special_tokens.size(); idx < vocab_size; ++idx) {
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

        int idx = BASE_VOCAB_SIZE + (int)special_tokens.size(), a, b;
        merges.clear();
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            iss >> a >> b;
            merges[{a, b}] = idx;
            idx++;
        }

        std::ifstream vocab_file(model_file + ".vocab");
        if (!vocab_file.is_open()) {
            std::cerr << "Error: Unable to open vocab file: " << model_file + ".vocab" << "\n";
            return false;
        }
        vocab.clear();
        while (std::getline(vocab_file, line)) {
            // Find the first '[' and the last ']' in the line
            size_t start = line.find('[');
            size_t end = line.rfind(']');  // Use rfind to find the last ']'
            if (start != std::string::npos && end != std::string::npos) {
                std::string token_str = line.substr(start + 1, end - start - 1);
                try {
                    idx = std::stoi(line.substr(end + 2));
                    vocab[idx] = string_to_bytes(token_str);
                }
                catch (const std::invalid_argument& /* e */) {
                    std::cerr << "Error: Invalid token ID in vocab file: " << line << "\n";
                    continue;
                }
            }
        }
        return true;
    }

    // Get the ID of a special token
    int get_special_token_id(const std::string& token) const {
        auto it = special_tokens.find(token);
        if (it != special_tokens.end()) return it->second;
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
    struct pair_hash {
        template <class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2>& p) const {
            auto hash1 = std::hash<T1>{}(p.first);
            auto hash2 = std::hash<T2>{}(p.second);
            return hash1 ^ (hash2 << 1);
        }
    };
    std::unordered_map<std::pair<int, int>, int, pair_hash> get_stats(const std::vector<int>& ids) {
        std::unordered_map<std::pair<int, int>, int, pair_hash> global_stats;
        std::mutex global_stats_mutex;

        auto worker = [&](size_t start, size_t end) {
            std::unordered_map<std::pair<int, int>, int, pair_hash> local_stats;
            for (size_t i = start; i < end - 1; ++i) {
                local_stats[{ids[i], ids[i + 1]}]++;
            }

            std::lock_guard<std::mutex> lock(global_stats_mutex);
            for (const auto& pair : local_stats) {
                global_stats[pair.first] += pair.second;
            }
            };

        size_t num_threads = std::thread::hardware_concurrency();
        size_t segment_size = ids.size() / num_threads;
        std::vector<std::thread> threads;

        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * segment_size;
            size_t end = (t == num_threads - 1) ? ids.size() : start + segment_size;
            threads.emplace_back(worker, start, end);
        }

        for (auto& thread : threads) thread.join();

        return global_stats;
    }

    // Finds the most frequent pair of tokens in the given statistics map that does not exceed the maximum token length
    std::pair<int, int> get_most_frequent_pair(const std::unordered_map<std::pair<int, int>, int, pair_hash>& stats) {
        std::pair<int, int> best_pair = { -1, -1 }; // Initialize the best pair to an invalid value
        int max_count = 0; // Initialize the maximum frequency count to 0

        // Iterate over all pairs in the statistics map
        for (const auto& stat : stats) {
            const std::pair<int, int>& pair = stat.first; // Extract the token pair
            int count = stat.second; // Extract the frequency count

            // Check if the new token formed by merging the pair would exceed the maximum allowed length
            size_t new_token_length = vocab[pair.first].size() + vocab[pair.second].size();
            if (new_token_length > MAX_TOKEN_LENGTH) {
                continue; // Skip this pair if it exceeds the maximum token length
            }

            // Update the best pair if the current pair has a higher frequency
            if (count > max_count) {
                best_pair = pair;
                max_count = count;
            }
        }

        return best_pair; // Return the most frequent valid pair
    }

    // Merge the most frequent pair in the token sequence
    std::vector<int> merge(std::vector<int>& ids, const std::pair<int, int>& pair, int idx) {
        std::vector<int> new_ids;
        new_ids.reserve(ids.size()); // Reserve space to avoid reallocations

        for (size_t i = 0; i < ids.size(); ++i) {
            if (i < ids.size() - 1 && ids[i] == pair.first && ids[i + 1] == pair.second) {
                new_ids.push_back(idx); // Replace the pair with the new token ID
                i++; // Skip the next token
            }
            else new_ids.push_back(ids[i]); // Keep the current token
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

// Function to build the training corpus with special tokens
std::vector<int> build_qa_tokens(const std::vector<std::pair<std::string, std::string>>& qa_pairs, bpe_tokenizer& tokenizer) {
    std::vector<int> tokens;

    for (const auto& qa_pair : qa_pairs) {
        // Encode the question and add the question token
        std::vector<int> q_tokens = tokenizer.encode(qa_pair.first);
        q_tokens.insert(q_tokens.begin(), tokenizer.get_special_token_id("<|question|>"));

        // Encode the response and add the response token
        std::vector<int> r_tokens = tokenizer.encode(qa_pair.second);
        r_tokens.insert(r_tokens.begin(), tokenizer.get_special_token_id("<|response|>"));
        r_tokens.push_back(tokenizer.get_special_token_id("<|endoftext|>"));

        // Combine the question and response tokens
        tokens.insert(tokens.end(), q_tokens.begin(), q_tokens.end());
        tokens.insert(tokens.end(), r_tokens.begin(), r_tokens.end());
    }

    return tokens;
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

std::string load_data_from_file_or_directory(const std::string& path, size_t max_size = 0.1 * 1024 * 1024) {
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
        parser.add_option("data-size", "Set the size of data to load in MB (default: 10 MB)", 1);
        parser.add_option("learning-rate", "Set the learning rate for training (default: 1e-4)", 1);
        parser.add_option("batch-size", "Set the mini-batch size for training (default: 64)", 1);
        parser.add_option("max-epochs", "Set the maximum number of training epochs (default: 100)", 1);
        parser.add_option("iterations-threshold", "Set the iterations without progress threshold (default: 15000)", 1);
        parser.add_option("min-learning-rate", "Set the minimum learning rate (default: 1e-6)", 1);
        parser.add_option("shuffle", "Shuffle training sequences and labels before training (default: false)");
        parser.add_option("temperature", "Set the temperature for text generation (default: 1.0)", 1);
        parser.add_option("top-k", "Set the number of top tokens to considere for response initialization (default: 1)", 1);
        parser.add_option("qna", "Enable Q&A mode for chatbot inference (default: false)");
        parser.parse(argc, argv);

        if (parser.number_of_arguments() == 0 && !parser.option("train")
            && !parser.option("chatbot")
            && !parser.option("test-tokenizer")) {
            std::cout << "Usage:\n"
                << "  --train: Train a Dlib transformer model on specific text\n"
                << "  --chatbot: Engage in an interactive chatbot session using a trained model\n"
                << "  --test-tokenizer: Test the BPE tokenizer\n"
                << "  --data <value>: Specify a file or directory containing the training data\n"
                << "  --data-size <value>: Set the size of data to load in MB (default: 10 MB)\n"
                << "  --learning-rate <value>: Set the learning rate for training (default: 1e-4)\n"
                << "  --batch-size <value>: Set the mini-batch size for training (default: 64)\n"
                << "  --max-epochs <value>: Set the maximum number of training epochs (default: 100)\n"
                << "  --iterations-threshold <value>: Set the iterations without progress threshold (default: 15000)\n"
                << "  --min-learning-rate <value>: Set the minimum learning rate (default: 1e-6)\n"
                << "  --shuffle: Shuffle training sequences and labels before training (default: false)\n"
                << "  --temperature <value>: Set the temperature for text generation (default: 1.0)\n"
                << "  --top-k <value>: Set the number of top tokens to considere for response initialization (default: 1)\n"
                << "  --qna: Enable Q&A mode for chatbot inference (default: false)\n";
            return 0;
        }

        // Test the tokenizer
        if (parser.option("test-tokenizer")) {
            bpe_tokenizer c_tokenizer;
            c_tokenizer.load("dlib_t3k_base");

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
        unsigned long batch_size = 64;
        long max_epochs = 100;
        unsigned long iterations_threshold = 15000;
        double min_learning_rate = 1e-6;
        double temperature = 1.0;
        unsigned long top_k = 1;
        unsigned long data_size = 10 * 1024 * 1024;

        // Override defaults if options are provided
        if (parser.option("learning-rate"))
            learning_rate = std::stod(parser.option("learning-rate").argument());
        if (parser.option("batch-size"))
            batch_size = std::stoul(parser.option("batch-size").argument());
        if (parser.option("max-epochs"))
            max_epochs = std::stoul(parser.option("max-epochs").argument());
        if (parser.option("iterations-threshold"))
            iterations_threshold = std::stoul(parser.option("iterations-threshold").argument());
        if (parser.option("min-learning-rate"))
            min_learning_rate = std::stod(parser.option("min-learning-rate").argument());
        if (parser.option("temperature"))
            temperature = std::stod(parser.option("temperature").argument());
        if (parser.option("top-k"))
            top_k = std::stoul(parser.option("top-k").argument());
        if (parser.option("data-size"))
            data_size = std::stoul(parser.option("data-size").argument()) * 1024 * 1024;

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
        const std::string model_file = "lm_chatbot_en_46M_fp32_model.dat";
        const std::string checkpoint_file = "checkpoint.dat";

        // Load the tokenizer
        bpe_tokenizer tokenizer;
        if (!tokenizer.load("dlib_t3k_base")) {
            std::cerr << "Failed to load vocabulary file (dlib_t3k_base.[model|vocab])!" << std::endl;
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
                training_data = load_data_from_file_or_directory(parser.option("data").argument(), data_size);            

            // 1) Tokenize the Shakespeare text
            std::cout << "Encoding sequences in progress...";
            std::vector<int> full_tokens = training_data.empty() ? build_qa_tokens(QA_PAIRS, tokenizer) : tokenizer.encode(training_data);
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
                    if (std::chrono::duration_cast<std::chrono::seconds>(current_time - last_print_time).count() >= 25) {
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
                            << "\tsteps w/o progress: " << trainer.get_steps_without_progress()
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
            bool qna_mode = parser.option("qna"); // Check if QnA mode is enabled
            std::cout << "=== CHATBOT MODE " << (qna_mode ? "(QnA mode activated) " : "") << "===\n";

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
                std::vector<int> user_tokens = tokenizer.encode(user_input);
                if (qna_mode) user_tokens.insert(user_tokens.begin(), tokenizer.get_special_token_id("<|question|>"));
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
                bool stop_generation = false, found_response_start = false;
                const size_t min_response_length = user_tokens.size();  // Minimum response length = user input length
                const size_t max_response_length = max_seq_len / 2;  // Maximum response length = half of context window
                size_t response_length = 0;
                float response_confidence = 1.0f;  // Confidence score for the generated response
                const size_t max_search_length = 20;  // Maximum tokens to search for a response start
                const float min_confidence_threshold = 0.7f;  // Minimum confidence threshold for a reliable response

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

                    // Update response confidence
                    response_confidence += std::log(logits(0, next_token));

                    // Decode the token to a string
                    std::string next_word = tokenizer.decode(next_token);
                    if (qna_mode) {
                        if (next_token == tokenizer.get_special_token_id("<|response|>")) {
                            found_response_start = true;
                        }
                        else if (next_token == tokenizer.get_special_token_id("<|endoftext|>")) {
                            stop_generation = true;
                        }
                    } else found_response_start = true;
                    if (found_response_start && !stop_generation) generated_text += next_word;
                    response_length++;

                    // Shift the input sequence to the left
                    for (long i = 0; i < max_seq_len - 1; ++i)
                        input_seq(i, 0) = input_seq(i + 1, 0);
                    input_seq(max_seq_len - 1, 0) = (int)next_token;

                    // Check for stopping conditions
                    if (qna_mode && !found_response_start && response_length >= max_search_length)
                        stop_generation = true;
                    if (response_length >= min_response_length) {                                              
                        // Stop if a sentence-ending punctuation mark is encountered
                        if (next_word.find_first_of(".!?") != std::string::npos ||
                            next_token == tokenizer.get_special_token_id("<|endoftext|>"))
                            stop_generation = true;
                    }

                    // Stop if the maximum response length is reached
                    if (response_length >= max_response_length)
                        stop_generation = true;
                }
                float n_confidence = std::exp(response_confidence / (response_length + 1));
                std::cout << "(confidence level: " << std::to_string(n_confidence) << ")" << std::endl;

                // Display the chatbot's response
                if (generated_text.empty() || n_confidence < min_confidence_threshold)
                    std::cout << "Chatbot: " << get_random_fallback_response() << std::flush << std::endl;
                else
                    std::cout << "Chatbot: " << generated_text << std::flush << std::endl;

                // Append the chatbot's response to the conversation history
                if (!generated_text.empty()) {
                    std::vector<int> response_tokens = tokenizer.encode(" " + generated_text + " ");
                    conversation_tokens.insert(conversation_tokens.end(), response_tokens.begin(), response_tokens.end());
                }

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