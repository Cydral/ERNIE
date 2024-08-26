/*!
 *  Copyright (c) 2024 by Cydral
 * \file advanced_tokenizer.cpp
 * \brief SentencePiece, TikToken, BERT, and HuggingFace tokenization methods (encode & decode)
 */

#include "advanced_tokenizer.hpp"
#include <dlib/base64.h>
#include <fstream>
#include <sstream>
#include <queue>
#include <functional>
#include <random>
#include <codecvt>
#include <regex>
#include <set>
#include <climits>

static std::string base64_decode(const std::string& str)
{
    dlib::base64 decoder;
    std::istringstream sin;
    std::ostringstream sout;    
    sin.str(str);
    decoder.decode(sin, sout);
    return sout.str();
}

static inline size_t one_char_len(const char *src)
{
  return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(*src & 0xFF) >> 4];
}

// ----------------------------------------------------------------------------------------

advanced_tokenizer* advanced_tokenizer::create_tokenizer(const std::string& filename)
{
    advanced_tokenizer* tokenizer = nullptr;
    std::ifstream tok_file(filename);
    DLIB_CASSERT(tok_file.good(), "Failed to load tokenizer from: " << filename);

    std::string line;
    std::getline(tok_file, line);
    std::istringstream line_str(line);
    int magic_number, tokenizer_type;
    line_str >> magic_number;
    DLIB_CASSERT(magic_number == MAGIC_NUMBER, "Invalid magic number in file: " << filename);
    line_str >> tokenizer_type;

    switch (tokenizer_type)
    {
    case SENTENCEPIECE:
        tokenizer = new sentencepiece_tokenizer();
        break;
    case TIKTOKEN:
        tokenizer = new tiktoken_tokenizer();
        break;
    case BERT:
        tokenizer = new bert_tokenizer();
        break;
    case HUGGINGFACE:
        tokenizer = new huggingface_tokenizer();
        break;
    default:
        DLIB_CASSERT(false, "Unknown tokenizer type: " << tokenizer_type);
    }

    tokenizer->load_special(tok_file);
    tokenizer->load_vocab(tok_file);
    tok_file.close();
    return tokenizer;
}

// ----------------------------------------------------------------------------------------

bool advanced_tokenizer::is_stop(int token) const
{
    return std::find(stop_tokens_.begin(), stop_tokens_.end(), token) != stop_tokens_.end();
}

// ----------------------------------------------------------------------------------------

bool advanced_tokenizer::is_special(int token) const
{
    return std::find(special_tokens_.begin(), special_tokens_.end(), token) != special_tokens_.end();
}

void advanced_tokenizer::load_special(std::ifstream& tok_file)
{
    std::string line;
    std::getline(tok_file, line);
    std::istringstream line_str(line);
    int special_num, stop_num, prefix_num;
    line_str >> special_num >> stop_num >> prefix_num;
    std::getline(tok_file, line);
    std::istringstream special_line(line);
    if (special_num) {
        special_tokens_.resize(special_num);
        for (int i = 0; i < special_num; i++) special_line >> special_tokens_[i];
    }
    if (stop_num) {
        stop_tokens_.resize(stop_num);
        for (int i = 0; i < stop_num; i++) special_line >> stop_tokens_[i];
    }
    if (prefix_num) {
        prefix_tokens_.resize(prefix_num);
        for (int i = 0; i < prefix_num; i++) special_line >> prefix_tokens_[i];
    }
}

// ----------------------------------------------------------------------------------------

std::vector<int> advanced_tokenizer::encode(const std::string& str)
{
    std::vector<int> ids = prefix_tokens_;
    if (!special_tokens_.empty()) {
        std::string text = str;
        size_t start = 0;
        for (size_t i = 0; i < text.length(); ++i) {
            for (auto special_id : special_tokens_) {
                const auto& token = decode(special_id);
                if (token.empty()) continue;
                if (i + token.length() <= text.length() && text.substr(i, token.length()) == token) {
                    if (i > start) encode(text.substr(start, i - start), ids);
                    ids.push_back(special_id);
                    start = i + token.length();
                    i = start - 1;
                    break;
                }
            }
        }
        if (start < text.length()) encode(text.substr(start), ids);
    } else {
        encode(str, ids);
    }
    return ids;
}

// ----------------------------------------------------------------------------------------

bool sentencepiece_tokenizer::load_vocab(std::ifstream& file)
{
    std::string line, token;
    std::getline(file, line);
    int vocab_len = std::stoi(line), type;
    float score;
    sentence_pieces_.resize(vocab_len);
    for (int index = 0; index < vocab_len; index++) {
        std::getline(file, line);
        std::istringstream line_str(line);
        line_str >> token >> score >> type;
        token = base64_decode(token);
        auto p_type = static_cast<piece_type>(type);
        sentence_piece piece = {token, score, p_type};
        sentence_pieces_[index] = std::move(piece);
        if (p_type == piece_type::NORMAL) {
            pieces_.insert({token, index});
        } else {
            reserved_id_map_.insert({token, index});
            if (p_type == piece_type::UNKNOWN) unk_id_ = index;
        }
    }
    return true;
}

// ----------------------------------------------------------------------------------------

int sentencepiece_tokenizer::piece_to_id(const std::string& piece) const
{
    auto it = reserved_id_map_.find(piece);
    if (it != reserved_id_map_.end()) return it->second;
    auto it2 = pieces_.find(piece);
    if (it2 != pieces_.end()) return it2->second;    
    return unk_id_;
}

// ----------------------------------------------------------------------------------------

std::string sentencepiece_tokenizer::byte_to_piece(unsigned char c) const
{
    const int len = ::snprintf(nullptr, 0, "<0x%02X>", c);
    std::string s;
    s.resize(len);
    ::snprintf(&s[0], s.size() + 1, "<0x%02X>", c);
    return s;
}

sentencepiece_tokenizer::encode_result sentencepiece_tokenizer::bpe_encode(string_view_ normalized, float alpha)
{
    struct symbol_pair {
        int left;
        int right;
        float score;
        size_t size;
    };

    struct symbol_pair_comparator {
        bool operator()(const symbol_pair* h1, const symbol_pair* h2) const {
            return (h1->score < h2->score || (h1->score == h2->score && h1->left > h2->left));
        }
    };

    struct symbol {
        int prev;
        int next;
        bool freeze = false;
        string_view_ piece;
    };

    using agenda_ = std::priority_queue<symbol_pair*, std::vector<symbol_pair*>, symbol_pair_comparator>;
    agenda_ agenda;
    std::vector<symbol> symbols;
    symbols.reserve(normalized.size());
    std::unordered_map<string_view_, std::pair<string_view_, string_view_>> rev_merge;
    std::vector<std::unique_ptr<symbol_pair>> symbol_pair_holder;

    // Lookup new symbol pair at [left, right] and inserts it to agenda
    auto maybe_add_new_symbol_pair = [this, &symbol_pair_holder, &symbols, &agenda, &rev_merge](int left, int right) {
        if (left == -1 || right == -1 || symbols[left].freeze || symbols[right].freeze) return;
        
        const string_view_ piece(symbols[left].piece.data(), symbols[left].piece.size() + symbols[right].piece.size());
        std::string piece_str(piece.to_string());
        const auto it = pieces_.find(piece_str);
        if (it == pieces_.end()) return;
        
        symbol_pair_holder.emplace_back(new symbol_pair);
        auto* h = symbol_pair_holder.back().get();
        h->left = left;
        h->right = right;
        h->score = get_score(it->second);
        h->size = piece.size();
        agenda.push(h);

        if (is_unused(it->second)) {
            rev_merge[piece] = std::make_pair(symbols[left].piece, symbols[right].piece);
        }
    };


    // Splits the input into character sequence
    int index = 0;
    while (!normalized.empty()) {
        symbol s;
        size_t mblen = std::min<size_t>(normalized.size(), one_char_len(normalized.data()));
        s.piece = string_view_(normalized.data(), mblen);
        s.prev = index == 0 ? -1 : index - 1;
        normalized.remove_prefix(mblen);
        s.next = normalized.empty() ? -1 : index + 1;
        ++index;
        symbols.emplace_back(s);
    }
    if (symbols.empty()) return {};
    
    // Lookup all bigrams
    for (size_t i = 1; i < symbols.size(); ++i) maybe_add_new_symbol_pair(i - 1, i);

    // BPE-dropout: https://arxiv.org/pdf/1910.13267.pdf
    std::mt19937 rand_gen;
    auto skip_merge = [&]() {
        if (alpha <= 0.0) return false;
        if (alpha >= 1.0) return true;
        std::uniform_real_distribution<> gen(0.0, 1.0);
        return gen(rand_gen) < alpha;
    };

    // Main loop
    while (!agenda.empty()) {
        symbol_pair* top = agenda.top();
        agenda.pop();

        // 'top' is no longer available
        if (symbols[top->left].piece.empty() || symbols[top->right].piece.empty() ||
            symbols[top->left].piece.size() + symbols[top->right].piece.size() != top->size) {
            continue;
        }

        if (skip_merge()) continue;
        // Replaces symbols with `top` rule
        symbols[top->left].piece = string_view_(
            symbols[top->left].piece.data(),
            symbols[top->left].piece.size() + symbols[top->right].piece.size());

        // Updates prev/next pointers
        symbols[top->left].next = symbols[top->right].next;
        if (symbols[top->right].next >= 0) {
            symbols[symbols[top->right].next].prev = top->left;
        }
        symbols[top->right].piece = string_view_("");

        // Adds new symbol pairs which are newly added after symbol replacement
        maybe_add_new_symbol_pair(symbols[top->left].prev, top->left);
        maybe_add_new_symbol_pair(top->left, symbols[top->left].next);
    }

    std::function<void(string_view_, encode_result*)> resegment;
    resegment = [this, &resegment, &rev_merge](string_view_ w, encode_result* output) {
        std::string w_str(w.to_string());
        const int id = piece_to_id(w_str);
        if (id == -1 || !is_unused(id)) {
            output->emplace_back(w, id);
            return;
        }
        const auto p = rev_merge.find(w);
        if (p == rev_merge.end()) {
            output->emplace_back(w, id);
            return;
        }
        resegment(p->second.first, output);
        resegment(p->second.second, output);
    };

    encode_result output;
    for (int index = 0; index != -1; index = symbols[index].next) {
        resegment(symbols[index].piece, &output);
    }
    return output;
}

// ----------------------------------------------------------------------------------------

void sentencepiece_tokenizer::encode(const std::string& str, std::vector<int>& ids)
{
    auto result = bpe_encode(str);
    for (const auto& p : result) {
        const string_view_ w = p.first;
        const int id = p.second;
        const bool is_unk = (id == unk_id_);
        if (is_unk && byte_fall_back_) {
            for (int i = 0; i < w.size(); ++i) {
                const char b = w[i];
                const auto piece = byte_to_piece(b);
                auto sp_id = piece_to_id(piece);
                ids.push_back(sp_id);
            }
        } else ids.push_back(id);
    }
}

// ----------------------------------------------------------------------------------------

std::string sentencepiece_tokenizer::decode(int id)
{
    DLIB_ASSERT(id >= 0 && id < sentence_pieces_.size(), "Invalid token ID");
    auto piece = sentence_pieces_[id].piece;
    size_t pos = piece.find("▁");
    if (pos != std::string::npos) piece.replace(pos, pos + 3, " ");
    return piece;
}

// ----------------------------------------------------------------------------------------

float sentencepiece_tokenizer::get_score(int id) const
{
    DLIB_ASSERT(id >= 0 && id < sentence_pieces_.size(), "Invalid token ID");
    return sentence_pieces_[id].score;
}

// ----------------------------------------------------------------------------------------

bool sentencepiece_tokenizer::is_unused(int id) const
{
    DLIB_ASSERT(id >= 0 && id < sentence_pieces_.size(), "Invalid token ID");
    return sentence_pieces_[id].type == piece_type::UNUSED;
}

// ----------------------------------------------------------------------------------------

bool sentencepiece_tokenizer::is_control(int id) const
{
    return sentence_pieces_[id].type == piece_type::CONTROL;
}

// ----------------------------------------------------------------------------------------

bool tiktoken_tokenizer::load_vocab(std::ifstream& tok_file)
{
    std::string line;
    std::getline(tok_file, line);
    int vocab_len = std::stoi(line);
    decoder_.resize(vocab_len);
    for (int i = 0; i < vocab_len; i++) {
        std::getline(tok_file, line);
        auto token = base64_decode(line);
        encoder_.insert({ token, i });
        decoder_[i] = token;
    }
    return true;
}

// ----------------------------------------------------------------------------------------

void tiktoken_tokenizer::encode(const std::string& str, std::vector<int>& ids)
{
    if (str.empty()) return;    
    size_t i = 0;
    while (i < str.size()) {
        bool found_pair = false;
        // Attempt to match the longest possible symbol
        size_t longest_match_len = 0;
        std::string longest_match;

        // Check substrings of decreasing length
        for (size_t len = str.size() - i; len > 0; --len) {
            std::string token = str.substr(i, len);
            auto it = encoder_.find(token);
            if (it != encoder_.end()) {
                if (len > longest_match_len) {
                    longest_match_len = len;
                    longest_match = it->first;
                }
            }
        }

        if (!longest_match.empty()) {
            ids.push_back(encoder_.at(longest_match));
            i += longest_match_len;
        } else {
            DLIB_CASSERT(false, "Error: No encoding found for the sequence starting at position " << i);
            return;
        }
    }
}

// ----------------------------------------------------------------------------------------

std::string tiktoken_tokenizer::decode(int id)
{
    DLIB_ASSERT(id >= 0 && id < decoder_.size(), "Invalid token ID");    
    return decoder_[id];
}

// ----------------------------------------------------------------------------------------

std::vector<int> bert_tokenizer::word_piece(const std::string& token)
{
    auto it = encoder_.find(token);
    if (it != encoder_.end()) return {it->second};
    
    std::vector<int> ids;
    std::string current = token;
    while (!current.empty()) {
        int match_id = -1;
        size_t match_pos = 0;
        for (int len = static_cast<int>(current.size()); len > 0; --len) {
            std::string candidate = current.substr(0, len);
            if (!ids.empty()) candidate = "##" + candidate;
            
            auto it = encoder_.find(candidate);
            if (it != encoder_.end()) {
                match_id = it->second;
                match_pos = len;
                break;
            }
        }
        if (match_id == -1) {
            // [UNK] token
            ids.push_back(100);
            break;
        }

        ids.push_back(match_id);
        // Not first word, adding ## prefix
        current = current.substr(match_pos);
    }
    return ids;
}

// ----------------------------------------------------------------------------------------

void bert_tokenizer::encode(const std::string& str, std::vector<int>& ids)
{
    std::vector<std::string> tokens;
    std::string current_token;
    size_t i = 0;
    while (i < str.size()) {
        current_token.clear();
        unsigned char c = static_cast<unsigned char>(str[i]);

        // Handle multi-byte UTF-8 characters
        if ((c & 0x80) != 0) {
            unsigned char mask = 0xE0; // 1110 0000 for 3-byte char
            if ((c & mask) == mask) {
                current_token = str.substr(i, 3);
                i += 3;
            } else {
                ++i;
                continue;
            }
        }
        // Handle continuous sequence of letters and digits
        else if (std::isalnum(c)) {
            while (i < str.size() && std::isalnum(static_cast<unsigned char>(str[i]))) {
                current_token += std::tolower(str[i]);
                ++i;
            }
        }
        // Handle punctuation and symbols
        else if (std::ispunct(c)) {
            current_token = str[i];
            ++i;
        }
        // Handle space, tab, enter
        else if (std::isspace(c)) {
            ++i;
            continue;
        }
        // Handle any other single-byte characters
        else {
            current_token = str[i];
            ++i;
        }
        if (!current_token.empty()) tokens.push_back(current_token);
    }

    for (const auto& token : tokens) {
        auto piece_ids = word_piece(token);
        ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
    }
}

std::wstring utf8_to_wstring(const std::string& str)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.from_bytes(str);
}

std::string wstring_to_utf8(const std::wstring& wstr)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.to_bytes(wstr);
}

// Given a token as a UTF8 string, encode each byte into an wchar_t
void byte_encode_token(const std::string& token,
    const std::unordered_map<uint8_t, wchar_t>& b2u,
    std::wstring* result)
{
    result->clear();
    for (unsigned char c : token) result->push_back(b2u.at(c));
}

// ----------------------------------------------------------------------------------------

bool huggingface_tokenizer::load_vocab(std::ifstream& tok_file)
{
    std::string line, token;
    int vocab_len, merge_len;
    std::getline(tok_file, line);
    std::istringstream line_str(line);
    line_str >> vocab_len >> merge_len;

    // Load vocab
    decoder_.resize(vocab_len);
    for (int i = 0; i < vocab_len; i++) {
        std::getline(tok_file, line);
        encoder_.insert({ line, i });
        decoder_[i] = line;
    }

    // Load merge_rule
    for (int i = 0; i < merge_len; i++) {
        std::getline(tok_file, line);
        size_t d = line.find(" ");
        DLIB_CASSERT(d != std::string::npos, "Invalid merge rule format");
        bpe_ranks_.insert({ {utf8_to_wstring(line.substr(0, d)),
                            utf8_to_wstring(line.substr(d + 1))}, i });
    }

    // bytes_to_unicode
    auto insert_range_ = [this](int start, int end) {
        for (int c = start; c <= end; c++) {
            b2u_.insert({ static_cast<uint8_t>(c), static_cast<wchar_t>(c) });
        }
    };

    b2u_.clear();
    insert_range_(0x0021, 0x007E);  // Unicode range for '!' to '~'
    insert_range_(0x00A1, 0x00AC);  // Unicode range for '¡' to '¬'
    insert_range_(0x00AE, 0x00FF);  // Unicode range for '®' to 'ÿ'

    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (b2u_.find(static_cast<uint8_t>(b)) == b2u_.end()) {
            b2u_.insert({ static_cast<uint8_t>(b), static_cast<wchar_t>(256 + n) });
            n++;
        }
    }
    for (const auto& e : b2u_) u2b_.insert({ e.second, e.first });
    
    return true;
}

// ----------------------------------------------------------------------------------------

void get_pairs(const std::wstring& word, std::vector<std::pair<std::wstring, std::wstring>>& pairs)
{
    pairs.clear();

    if (word.size() < 2) return;

    wchar_t previous = word[0];
    for (size_t i = 1; i < word.size(); i++) {
        pairs.push_back({ std::wstring(1, previous), std::wstring(1, word[i]) });
        previous = word[i];
    }
}

// ----------------------------------------------------------------------------------------

void huggingface_tokenizer::bpe(const std::wstring& token, const bpe_ranks& bpe_ranks, std::vector<std::wstring>& result)
{
    std::set<int> merged;  // Records indices in pairs that were merged
    auto left = [&merged](int i) {
        for (int j = i - 1; j >= -1; j--) {
            if (merged.find(j) == merged.end()) return j;
        }
        return -1;
    };
    auto right = [&merged](int i, int cap) {
        for (int j = i + 1; j < cap; j++) {
            if (merged.find(j) == merged.end()) return j;
        }
        return cap;
    };

    std::vector<std::pair<std::wstring, std::wstring>> pairs;
    get_pairs(token, pairs);

    while (true) {
        int min_score = INT_MAX;
        int to_merge = -1;  // Indices into pairs

        for (size_t i = 0; i < pairs.size(); ++i) {
            if (merged.find(i) == merged.end()) {  // Pair i is not merged
                auto iter = bpe_ranks.find(pairs[i]);
                int score = iter != bpe_ranks.end() ? iter->second : INT_MAX;
                if (score < min_score) {
                    min_score = score;
                    to_merge = i;
                }
            }
        }

        if (to_merge == -1) break;

        merged.insert(to_merge);
        std::wstring merge_into = pairs[to_merge].first + pairs[to_merge].second;

        int l = left(to_merge);
        if (l >= 0) pairs[l].second = merge_into;
        int r = right(to_merge, pairs.size());
        if (r < static_cast<int>(pairs.size())) pairs[r].first = merge_into;
    }

    if (merged.size() == pairs.size()) {
        result.push_back(token);
    } else {
        for (size_t i = 0; i < pairs.size(); ++i) {
            if (merged.find(i) == merged.end()) {
                if (left(i) < 0) result.push_back(pairs[i].first);
                result.push_back(pairs[i].second);
            }
        }
    }
}

// ----------------------------------------------------------------------------------------

void huggingface_tokenizer::encode(const std::string& str, std::vector<int>& ids)
{
    std::regex re("('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\\s\\w]+|\\s+)");
    std::string input = str;
    std::vector<std::string> result;
    std::string token;
    std::smatch match;

    while (std::regex_search(input, match, re)) {
        token = match.str(0);
        input = match.suffix().str();
        std::wstring wtoken;
        byte_encode_token(token, b2u_, &wtoken);        

        std::vector<std::wstring> bpe_tokens;
        bpe(wtoken, bpe_ranks_, bpe_tokens);

        for (auto ws : bpe_tokens) result.push_back(wstring_to_utf8(ws));
    }
    for (auto s : result) {
        auto it = encoder_.find(s);
        DLIB_CASSERT(it != encoder_.end(), "Token not found in encoder: " << s);
        ids.push_back(it->second);
    }
}

// ----------------------------------------------------------------------------------------

std::string huggingface_tokenizer::decode(int id)
{
    DLIB_CASSERT(id >= 0 && id < static_cast<int>(decoder_.size()), "Invalid token ID");
    if (id >= decoder_.size()) return "";
    std::wstring w = utf8_to_wstring(decoder_.at(id));
    std::string r;
    for (wchar_t c : w) {
        auto it = u2b_.find(c);
        DLIB_CASSERT(it != u2b_.end(), "Character not found in u2b_ map");
        r.push_back(static_cast<char>(it->second));
    }
    return r;
}
