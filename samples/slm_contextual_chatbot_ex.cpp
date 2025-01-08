#include <iostream>
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

const size_t VOCAB_SIZE = 5000;

// ----------------------------------------------------------------------------------------
class bpe {
public:
    static const std::string REGEX_PATTERN;

    explicit bpe(size_t vocab_size) : target_vocab_size_(vocab_size) {}

    void learn(const std::string& corpus) {
        // Tokenize the corpus using the regex pattern
        std::regex regex_pattern(REGEX_PATTERN, std::regex_constants::icase);
        std::vector<std::string> tokens;
        auto words_begin = std::sregex_iterator(corpus.begin(), corpus.end(), regex_pattern);
        auto words_end = std::sregex_iterator();

        for (std::sregex_iterator it = words_begin; it != words_end; ++it)
            tokens.push_back(it->str());

        // Initialize token frequencies and vocabulary
        std::unordered_map<std::string, int> token_frequencies;
        std::set<std::string> initial_vocab;
        for (const std::string& token : tokens) {
            if (token.substr(0, 2) == "<|" && token.substr(token.length() - 2) == "|>") {
                token_frequencies[token]++;
                initial_vocab.insert(token);
            } else {
                token_frequencies[token]++;
                for (char c : token) initial_vocab.insert(std::string(1, c));
            }
        }

        // Check if the requested vocab size is smaller than the initial vocab
        if (target_vocab_size_ <= initial_vocab.size()) {
            vocab_ = std::vector<std::string>(initial_vocab.begin(), initial_vocab.end());
            vocab_.resize(target_vocab_size_);
            std::cout << "Warning: Requested vocabulary size is smaller than or equal to the initial character set." << std::endl;
            std::cout << "Final vocabulary size: " << vocab_.size() << std::endl;
            return;
        }

        std::unordered_map<std::string, int> vocab;
        for (const auto& c : initial_vocab) vocab[c] = 0;  // Initialize with 0 frequency, will be updated later

        // Initialize pair frequencies
        std::unordered_map<std::pair<std::string, std::string>, int, PairHash> pair_frequencies;
        for (const auto& entry : token_frequencies) {
            const std::string& token = entry.first;
            std::vector<std::string> parts;
            for (char c : token) parts.push_back(std::string(1, c));
            update_pair_frequencies(parts, entry.second, pair_frequencies);
        }

        // Main BPE loop
        size_t prev_vocab_size = vocab.size();
        size_t stagnant_iterations = 0;
        const size_t max_stagnant_iterations = 5;

        while (vocab.size() < target_vocab_size_) {
            // Find the most frequent pair
            auto best_pair = std::max_element(
                pair_frequencies.begin(), pair_frequencies.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; }
            );

            if (best_pair == pair_frequencies.end()) {
                std::cout << "\nWarning: No more pairs to merge. Vocabulary size might be smaller than requested." << std::endl;
                break;
            }

            std::string new_token = best_pair->first.first + best_pair->first.second;
            vocab[new_token] = best_pair->second;

            // Update tokens and frequencies
            for (auto& entry : token_frequencies) {
                std::string token = entry.first;
                int freq = entry.second;
                std::vector<std::string> new_parts;
                std::vector<std::string> parts;
                for (char c : token) parts.push_back(std::string(1, c));

                for (size_t i = 0; i < parts.size(); ++i) {
                    if (i < parts.size() - 1 && parts[i] == best_pair->first.first
                        && parts[i + 1] == best_pair->first.second) {
                        new_parts.push_back(new_token);
                        ++i;
                    } else new_parts.push_back(parts[i]);
                }

                // Update pair frequencies
                if (parts != new_parts) {
                    update_pair_frequencies(parts, -freq, pair_frequencies);
                    update_pair_frequencies(new_parts, freq, pair_frequencies);
                    token = join(new_parts);
                }
            }

            // Remove the merged pair from pair frequencies
            pair_frequencies.erase(best_pair->first);

            // Check for stagnation
            if (vocab.size() - prev_vocab_size < 3) {
                stagnant_iterations++;
                if (stagnant_iterations >= max_stagnant_iterations) {
                    std::cout << "\nWarning: Vocabulary size stagnated. Stopping early." << std::endl;
                    break;
                }
            } else stagnant_iterations = 0;
            prev_vocab_size = vocab.size();
        }

        // Build the final vocabulary
        vocab_.clear();
        for (const auto& entry : vocab) vocab_.push_back(entry.first);

        // If we still haven't reached the desired vocab size, add the most frequent tokens
        if (vocab_.size() < target_vocab_size_) {
            std::vector<std::pair<std::string, int>> sorted_tokens(token_frequencies.begin(), token_frequencies.end());
            std::sort(sorted_tokens.begin(), sorted_tokens.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });

            for (const auto& token : sorted_tokens) {
                if (std::find(vocab_.begin(), vocab_.end(), token.first) == vocab_.end()) {
                    vocab_.push_back(token.first);
                    if (vocab_.size() == target_vocab_size_) break;
                }
            }
        }

        // Update the actual vocabulary size
        target_vocab_size_ = vocab_.size();
        std::cout << "\nFinal vocabulary size: " << target_vocab_size_ << std::endl;
    }

    const std::vector<std::string>& get_vocab() const {
        return vocab_;
    }

private:
    // Hash function for pairs
    struct PairHash {
        size_t operator()(const std::pair<std::string, std::string>& p) const {
            return std::hash<std::string>()(p.first) ^ std::hash<std::string>()(p.second);
        }
    };

    void update_pair_frequencies(const std::vector<std::string>& parts, int freq,
        std::unordered_map<std::pair<std::string, std::string>, int, PairHash>& pair_frequencies) {
        for (size_t i = 0; i < parts.size() - 1; ++i) {
            std::pair<std::string, std::string> pair = { parts[i], parts[i + 1] };
            pair_frequencies[pair] += freq;
            if (pair_frequencies[pair] <= 0) pair_frequencies.erase(pair);
        }
    }

    std::string join(const std::vector<std::string>& parts) {
        std::string result;
        for (const auto& part : parts) result += part;
        return result;
    }

    size_t target_vocab_size_;
    std::vector<std::string> vocab_;
};

// Define the regex pattern
const std::string bpe::REGEX_PATTERN =
    R"(<\|[^|]+\|>|'s|'t|'re|'ve|'m|'ll|'d|[[:alpha:]]+|[[:digit:]]+|\s+|\S)";

class tiktoken_tokenizer {
public:
    // Constructor initializes special token IDs to -1 (invalid)
    tiktoken_tokenizer() : unknown_token_id_(-1), pad_token_id_(-1) {}

    // Load vocabulary from a file
    bool load_vocab(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Failed to open vocab file: " << filepath << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            size_t space_pos = line.find(' ');
            if (space_pos == std::string::npos) continue; // Skip malformed lines

            // Extract token (base64) and ID
            std::string base64_token = line.substr(0, space_pos);
            int id = std::stoi(line.substr(space_pos + 1));

            // Decode base64 token to string
            std::string token = base64_decode(base64_token);

            // Store mapping
            encoder_[token] = id;
            if (id >= decoder_.size()) decoder_.resize(id + 1);
            decoder_[id] = token;
            if (token == "<|unknown|>") unknown_token_id_ = id;
            else if (token == "<|padding|>") pad_token_id_ = id;
        }

        return true;
    }

    // Learn BPE vocabulary from a text corpus
    void learn_bpe(const std::string& corpus, size_t vocab_size) {
        std::vector<std::string> special_tokens = {
            "<|unknown|>", "<|padding|>", "<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>",
            "<|fim_suffix|>", "<|endofprompt|>", "<|im_start|>", "<|im_end|>"
        };

        bpe c_bpe(vocab_size - special_tokens.size());
        c_bpe.learn(corpus);
        const std::vector<std::string>& vocab = c_bpe.get_vocab();

        // Build the vocabulary
        for (size_t i = 0; i < vocab.size(); ++i) {
            encoder_[vocab[i]] = i;
            if (i >= decoder_.size()) decoder_.resize(i + 1);
            decoder_[i] = vocab[i];
        }

        // Add special tokens
        for (const auto& token : special_tokens) {
            if (encoder_.find(token) == encoder_.end()) {
                size_t token_id = decoder_.size();
                decoder_.push_back(token);
                encoder_[token] = token_id;
            }
        }
        unknown_token_id_ = encoder_["<|unknown|>"];
        pad_token_id_ = encoder_["<|padding|>"];
    }

    // Save the learned vocabulary to a file
    bool save_vocab(const std::string& filepath) const {
        std::ofstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Failed to open vocab file for writing: " << filepath << std::endl;
            return false;
        }

        for (const auto& entry : encoder_) {
            std::string base64_token = base64_encode(entry.first);
            file << base64_token << " " << entry.second << "\n";
        }

        return true;
    }

    // Encode a string into a sequence of token IDs
    std::vector<int> encode(const std::string& str) const {
        std::vector<int> ids;
        if (str.empty()) return ids;

        std::regex regex_pattern(bpe::REGEX_PATTERN, std::regex_constants::icase);
        auto words_begin = std::sregex_iterator(str.begin(), str.end(), regex_pattern);
        auto words_end = std::sregex_iterator();

        for (std::sregex_iterator it = words_begin; it != words_end; ++it) {
            std::string chunk = it->str();

            // Check if the chunk is a special token
            auto special_token_it = encoder_.find(chunk);
            if (special_token_it != encoder_.end()) {
                ids.push_back(special_token_it->second);
                continue;
            }

            // Tokenize non-special tokens
            size_t start = 0;
            while (start < chunk.length()) {
                size_t end = chunk.length();
                bool found = false;
                while (end > start && !found) {
                    std::string substr = chunk.substr(start, end - start);
                    auto it = encoder_.find(substr);
                    if (it != encoder_.end()) {
                        ids.push_back(it->second);
                        found = true;
                    } else end--;
                }
                if (!found) {
                    // If no token is found, treat the character as unknown
                    ids.push_back(unknown_token_id_);
                    start++;
                } else start = end;
            }
        }

        return ids;
    }

    // Decode a token ID into a string
    std::string decode(int id) const {
        return (id < 0 || id >= decoder_.size()) ? "<|unknown|>" : decoder_[id];
    }

    // Get the total vocabulary size (including special tokens)
    size_t get_vocab_size() const { return decoder_.size(); }

    // Get the ID of the padding token
    int get_pad_token_id() const { return pad_token_id_; }

private:
    // Decode a base64 string to a string
    static std::string base64_decode(const std::string& base64_str) {
        dlib::base64 decoder;
        std::istringstream sin(base64_str);
        std::ostringstream sout;
        decoder.decode(sin, sout);
        return sout.str();
    }

    // Encode a string into base64
    static std::string base64_encode(const std::string& str) {
        dlib::base64 encoder;
        std::ostringstream sout;
        std::istringstream sin(str);
        encoder.encode(sin, sout);
        return sout.str();
    }

    std::unordered_map<std::string, int> encoder_; // string token -> ID
    std::vector<std::string> decoder_;             // ID -> string token
    int unknown_token_id_;                         // ID for unknown tokens
    int pad_token_id_;                             // ID for padding tokens
};

// ----------------------------------------------------------------------------------------

std::string load_data_from_file_or_directory(const std::string& path) {
    std::string data;

    try {
        dlib::directory dir(path);
        std::vector<dlib::file> files = dir.get_files();
        for (const auto& file : files) {
            std::ifstream input(file.full_name());
            if (input.is_open()) {
                std::cout << "Loading file: " << file.full_name() << std::endl;
                std::string content((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
                if (!data.empty()) data += "\n\n";
                data += content;
            }            
        }
    }
    catch (const dlib::directory::dir_not_found) {
        dlib::file file(path);
        std::ifstream input(file.full_name());
        if (input.is_open()) {
            std::cout << "Loading file: " << file.full_name() << std::endl;
            data.assign((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        }            
    }
        
    return data;
}

// ----------------------------------------------------------------------------------------

// Function to shuffle samples and labels in sync
void shuffle_samples_and_labels(std::vector<dlib::matrix<int, 0, 1>>& samples, std::vector<unsigned long>& labels) {
    std::vector<size_t> indices(samples.size());
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ..., N-1
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine{});

    // Create temporary vectors to hold shuffled data
    std::vector<dlib::matrix<int, 0, 1>> shuffled_samples(samples.size());
    std::vector<unsigned long> shuffled_labels(labels.size());

    // Apply the shuffle
    for (size_t i = 0; i < indices.size(); ++i) {
        shuffled_samples[i] = samples[indices[i]];
        shuffled_labels[i] = labels[indices[i]];
    }

    // Replace the original data with shuffled data
    samples = std::move(shuffled_samples);
    labels = std::move(shuffled_labels);
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) {
    try {
        dlib::command_line_parser parser;
        parser.add_option("train", "Train a small transformer on the built-in Shakespeare text");
        parser.add_option("chatbot", "Engage in an interactive chatbot session using a trained model");
        parser.add_option("train-tokenizer", "Train the TikToken tokenizer");
        parser.add_option("data", "Specify a file or directory containing the training data", 1);
        parser.add_option("learning-rate", "Set the learning rate for training (default: 1e-4)", 1);
        parser.add_option("batch-size", "Set the mini-batch size for training (default: 64)", 1);
        parser.add_option("max-epochs", "Set the maximum number of training epochs (default: 100)", 1);
        parser.add_option("iterations-threshold", "Set the iterations without progress threshold (default: 5000)", 1);
        parser.add_option("min-learning-rate", "Set the minimum learning rate (default: 1e-6)", 1);
        parser.add_option("shuffle", "Shuffle training sequences and labels before training (default: false)");
        parser.parse(argc, argv);

        if (parser.number_of_arguments() == 0 && !parser.option("train")
            && !parser.option("chatbot")
            && !parser.option("train-tokenizer")) {
            std::cout << "Usage:\n"
                << "  --train    : Train a small transformer model on the Shakespeare text\n"
                << "  --chatbot  : Engage in an interactive chatbot session using a trained model\n"
                << "  --train-tokenizer : Train the TikToken-like tokenizer\n"
                << "  --data <path>    : Specify a file or directory containing the training data\n"
                << "  --learning-rate <value> : Set the learning rate for training (default: 1e-4)\n"
                << "  --batch-size <value>    : Set the mini-batch size for training (default: 64)\n"
                << "  --max-epochs <value>    : Set the maximum number of training epochs (default: 100)\n"
                << "  --iterations-threshold <value> : Set the iterations without progress threshold (default: 5000)\n"
                << "  --min-learning-rate <value> : Set the minimum learning rate (default: 1e-6)\n"
                << "  --shuffle               : Shuffle training sequences and labels before training (default: false)\n";
            return 0;
        }

        // Load the tokenizer
        tiktoken_tokenizer tokenizer;
        if (!tokenizer.load_vocab("r5k_base.tiktoken")) {
            std::cerr << "Failed to load vocabulary file!" << std::endl;
            return 1;
        }

        // Test the tokenizer
        if (parser.option("train-tokenizer")) {
            tiktoken_tokenizer c_tokenizer;

            // Learn a new vocabulary from a corpus
            std::string corpus = parser.option("data") ? load_data_from_file_or_directory(parser.option("data").argument()) : "";
            if (corpus.empty()) return 0;
            c_tokenizer.learn_bpe(corpus, VOCAB_SIZE);
            c_tokenizer.save_vocab("r5k_base.tiktoken");

            // Test the tokenizer
            std::string input = "<|endoftext|>The quick brown fox jumps over the lazy dog, and the dog barks loudly!<|endoftext|>";
            std::vector<int> encoded = c_tokenizer.encode(input);
            std::cout << "Encoded tokens: ";
            for (int id : encoded) std::cout << id << " ";
            std::cout << std::endl;

            std::string decoded;
            for (int id : encoded) decoded += c_tokenizer.decode(id);
            std::cout << "Decoded string: " << decoded << std::endl;

            std::cout << "Total vocabulary size (including special tokens): "
                << c_tokenizer.get_vocab_size() << std::endl;

            return 0;
        }

        // Default values
        double learning_rate = 1e-4;
        long batch_size = 64;
        int max_epochs = 100;
        int iterations_threshold = 5000;
        double min_learning_rate = 1e-6;

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

        // Minimal configiguration for our Transformer mmodel
        const long vocab_size = VOCAB_SIZE;
        const long num_layers = 6;
        const long num_heads = 4;
        const long embedding_dim = 64;
        const long max_seq_len = 200;
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

        // The model file to store or load
        const std::string model_file = "lm_tiktoken_50k_fp32_model.dat";

        // ----------------------------------------------------------------------------------------
        // Train mode
        // ----------------------------------------------------------------------------------------
        if (parser.option("train"))
        {
            std::cout << "=== TRAIN MODE ===\n";

            // Load data from the specified file or directory
            std::string training_data;
            if (parser.option("data"))
                training_data = load_data_from_file_or_directory(parser.option("data").argument());
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
            std::cout << "Training text size: " << full_tokens.size() << " tokens\n";
            std::cout << "Maximum number of sequences: " << max_sequences << "\n";

            // Check if the text is too short
            if (max_sequences == 0) return 0;

            std::vector<dlib::matrix<int, 0, 1>> samples;
            std::vector<unsigned long> labels;

            // Let's create a training set of about (N) samples from the text
            const size_t N = max_sequences; // Use all available sequences
            for (size_t start = 0; start < N; ++start)
            {
                dlib::matrix<int, 0, 1> seq(max_seq_len, 1);
                for (long t = 0; t < max_seq_len; ++t)
                    seq(t, 0) = full_tokens[start + t];
                samples.push_back(seq);
                labels.push_back(full_tokens[start + max_seq_len]);
            }

            // Shuffle samples and labels if the --shuffle option is enabled
            if (parser.option("shuffle"))
            {
                std::cout << "Shuffling training sequences and labels...";
                shuffle_samples_and_labels(samples, labels);
                std::cout << " done\n";
            }

            // 2) Construct the network in training mode
            using net_type = my_transformer_cfg::network_type<true>;
            net_type net;
            if (dlib::file_exists(model_file))
            {
                std::cout << "Loading existing model...";
                dlib::deserialize(model_file) >> net;
                std::cout << " done\n";
            }

            // 3) Create dnn_trainer
            dlib::dnn_trainer<net_type, dlib::adam> trainer(net, dlib::adam(learning_rate, 0.9, 0.999), gpus);
            trainer.set_learning_rate(learning_rate);
            trainer.set_min_learning_rate(min_learning_rate);
            trainer.set_mini_batch_size(batch_size);
            trainer.set_iterations_without_progress_threshold(iterations_threshold);
            trainer.set_max_num_epochs(max_epochs);
            trainer.be_verbose();

            // 4) Train
            trainer.train(samples, labels);

            // 5) Evaluate quickly on the training set
            auto predicted = net(samples);
            size_t correct = 0;
            for (size_t i = 0; i < labels.size(); ++i)
                if (predicted[i] == labels[i])
                    correct++;
            double accuracy = (double)correct / labels.size();
            std::cout << "Training accuracy (on this sample set): " << accuracy << "\n";

            // 6) Save the model
            net.clean();
            dlib::serialize(model_file) << net;
            std::cout << "Model saved to " << model_file << "\n";
        }

        // ----------------------------------------------------------------------------------------
        // Chatbot mode
        // ----------------------------------------------------------------------------------------
        if (parser.option("chatbot"))
        {
            std::cout << "=== CHATBOT MODE ===\n";

            // 1) Load the trained model
            using net_infer = my_transformer_cfg::network_type<false>;
            net_infer net;
            if (dlib::file_exists(model_file))
            {
                dlib::deserialize(model_file) >> net;
                std::cout << "Loaded model from " << model_file << "\n";
            }
            else
            {
                std::cerr << "Error: model file not found. Please run --train first.\n";
                return 0;
            }
            std::cout << my_transformer_cfg::model_info::describe() << std::endl;
            std::cout << "Model parameters: " << count_parameters(net) << std::endl << std::endl;

            // 2) Initialize the conversation history
            std::vector<int> conversation_tokens;

            // 3) Conversation loop
            std::string user_input;
            while (true)
            {
                // Prompt the user for input
                std::cout << "You: ";
                std::getline(std::cin, user_input);

                // Exit if the user types "bye"
                if (user_input == "bye")
                {
                    std::cout << "Chatbot: Goodbye!\n";
                    break;
                }

                // Append the user's input to the conversation history
                std::vector<int> user_tokens = tokenizer.encode(user_input + "\n");
                conversation_tokens.insert(conversation_tokens.end(), user_tokens.begin(), user_tokens.end());

                // Truncate the conversation history to fit within max_seq_len
                if (conversation_tokens.size() > (size_t)max_seq_len)
                {
                    conversation_tokens.erase(conversation_tokens.begin(), conversation_tokens.end() - max_seq_len);
                }

                // Create a dlib matrix for the model input
                dlib::matrix<int, 0, 1> input_seq(max_seq_len, 1);
                for (long i = 0; i < max_seq_len; ++i)
                {
                    if ((size_t)i < conversation_tokens.size())
                        input_seq(i, 0) = conversation_tokens[i];
                    else
                        input_seq(i, 0) = tokenizer.get_pad_token_id(); // Use the padding token ID
                }

                // Generate a response
                std::string generated_text;
                bool stop_generation = false;
                const size_t min_response_length = user_tokens.size(); // Minimum response length = user input length
                const size_t max_response_length = max_seq_len / 2;    // Maximum response length = half of context window
                size_t response_length = 0;

                while (!stop_generation)
                {
                    unsigned long next_token = net(input_seq); // Single inference

                    // Decode the token to a string
                    std::string next_word = tokenizer.decode(next_token);
                    generated_text += next_word;
                    response_length++;

                    // Shift the input sequence to the left
                    for (long i = 0; i < max_seq_len - 1; ++i)
                        input_seq(i, 0) = input_seq(i + 1, 0);
                    input_seq(max_seq_len - 1, 0) = (int)next_token;

                    // Check for stopping conditions
                    if (response_length >= min_response_length)
                    {
                        // Stop if a sentence-ending punctuation mark is encountered
                        if (next_word.find_first_of(".!?") != std::string::npos)
                            stop_generation = true;
                    }

                    // Stop if the maximum response length is reached
                    if (response_length >= max_response_length)
                        stop_generation = true;
                }

                // Display the chatbot's response
                std::cout << "Chatbot: " << generated_text << "\n";

                // Append the chatbot's response to the conversation history
                std::vector<int> response_tokens = tokenizer.encode(generated_text + "\n");
                conversation_tokens.insert(conversation_tokens.end(), response_tokens.begin(), response_tokens.end());

                // Truncate the conversation history again to fit within max_seq_len
                if (conversation_tokens.size() > (size_t)max_seq_len)
                {
                    conversation_tokens.erase(conversation_tokens.begin(), conversation_tokens.end() - max_seq_len);
                }
            }
        }

        return 0;
    }
    catch (std::exception& e)
    {
        std::cerr << "Exception thrown: " << e.what() << std::endl;
        return 1;
    }
}