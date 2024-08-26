#ifndef TOKENIZER_hpp
#define TOKENIZER_hpp

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <iostream>
#include <cstring>

// ----------------------------------------------------------------------------------------

class string_view_
{
    /*!
        WHAT THIS OBJECT REPRESENTS
            This class represents a non-owning reference to a string or a substring.
            It's a lightweight alternative to std::string_view for environments where
            C++17 features are not available.

            This implementation provides basic functionalities similar to std::string_view,
            including:
                - Non-owning views of strings
                - Constant time operations for most methods
                - No memory allocations

            IMPORTANT:
                If you are using a C++17 compliant compiler or newer, it is recommended
                to use std::string_view instead. std::string_view offers better optimization,
                more comprehensive functionality, and better integration with the C++
                standard library.

        THREAD SAFETY
            It is safe to access const methods of this object from multiple threads.
            However, any operation that modifies the object is not thread-safe.
    !*/

public:
    string_view_() : data_(nullptr), size_(0) {}
    string_view_(const char* data) : data_(data), size_(data ? std::strlen(data) : 0) {}
    string_view_(const char* data, std::size_t size) : data_(data), size_(size) {}
    string_view_(const std::string& str) : data_(str.data()), size_(str.size()) {}
    constexpr string_view_(const string_view_&) noexcept = default;
    string_view_& operator=(const string_view_&) noexcept = default;

    const char& operator[](size_t pos) const
    {
        if (pos >= size_) throw std::out_of_range("Index out of range");
        return data_[pos];
    }
    constexpr const char* data() const noexcept { return data_; }
    constexpr std::size_t size() const noexcept { return size_; }
    constexpr bool empty() const noexcept { return size_ == 0; }
    std::string to_string() const { return std::string(data_, size_); }

    bool operator==(const string_view_& other) const noexcept
    {
        return size_ == other.size_ && (data_ == other.data_ || std::memcmp(data_, other.data_, size_) == 0);
    }
    bool operator!=(const string_view_& other) const noexcept
    {
        return !(*this == other);
    }

    void remove_prefix(size_t n)
    {
        n = std::min(n, size_);
        data_ += n;
        size_ -= n;
    }

    const char* begin() const noexcept { return data_; }
    const char* end() const noexcept { return data_ + size_; }

    int compare(string_view_ s) const noexcept
    {
        const size_t rlen = std::min(size_, s.size_);
        int result = std::memcmp(data_, s.data_, rlen);
        if (result == 0) {
            if (size_ < s.size_) result = -1;
            else if (size_ > s.size_) result = 1;
        }
        return result;
    }

private:
    const char* data_;
    std::size_t size_;
};

namespace std {
    template<>
    class hash<string_view_> {
    public:
        size_t operator()(const string_view_& sv) const {
            size_t result = 0;
            for (size_t i = 0; i < sv.size(); ++i) {
                result = (result * 31) + static_cast<size_t>(sv[i]);
            }
            return result;
        }
    };
}

class Tokenizer {
public:
    static constexpr int MAGIC_NUMBER = 430;
    enum TokenizerType {
        SENTENCEPIECE = 0,
        TIKTOKEN = 1,
        BERT = 2,
        HUGGINGFACE = 3
    };
    Tokenizer() = default;
    virtual ~Tokenizer() = default;
    static Tokenizer* createTokenizer(const std::string& filename);
    bool is_stop(int token);
    bool is_special(int token);
    std::vector<int> encode(const std::string& str);
    virtual std::string decode(int id) = 0;
protected:
    virtual void load_special(std::ifstream& file);
    virtual bool load_vocab(std::ifstream& file) = 0;
    virtual void encode(const std::string& str, std::vector<int>& ids) = 0;
    std::vector<int> special_tokens_;
    std::vector<int> stop_tokens_;
    std::vector<int> prefix_tokens_;
};

class Sentencepiece : public Tokenizer {
public:
    Sentencepiece() = default;
    virtual std::string decode(int id) override;
protected:
    virtual bool load_vocab(std::ifstream& file) override;
    virtual void encode(const std::string& str, std::vector<int>& ids) override;
private:
    enum ModelType {
        UNIGRAM = 1,
        BPE = 2,
        WORD = 3,
        CHAR = 4
    };
    enum PieceType {
        NORMAL = 1,
        UNKNOWN = 2,
        CONTROL = 3,
        USER_DEFINED = 4,
        UNUSED = 5,
        BYTE = 6
    };
    struct SentencePiece {
        std::string piece;
        float score;
        PieceType type = PieceType::NORMAL;
        SentencePiece() {}
        SentencePiece(const std::string& p, float s, PieceType t) : piece(p), score(s), type(t) {}
    };
    using EncodeResult = std::vector<std::pair<string_view_, int>>;
private:
    // model train type
    ModelType type_ = BPE;
    // byte fall back enable
    bool byte_fall_back_ = true;
    // unknown id.
    int unk_id_ = 0;
    // pieces from model
    std::vector<SentencePiece> sentence_pieces_;
    // piece -> id map for normal pieces
    std::unordered_map<std::string, int> pieces_;
    // piece -> id map for control, unknown, and byte pieces
    std::unordered_map<std::string, int> reserved_id_map_;
private:
    float get_score(int id) const;
    bool is_unused(int id) const;
    bool is_control(int id) const;
    int piece_to_id(const std::string& w) const;
    std::string byte_to_piece(unsigned char c) const;
    EncodeResult bpe_encode(string_view_ str, float alpha = 0.f);
};

class Tiktoken : public Tokenizer {
public:
    Tiktoken() = default;
    virtual std::string decode(int id) override;
protected:
    virtual bool load_vocab(std::ifstream& file) override;
    virtual void encode(const std::string& str, std::vector<int>& ids) override;
    std::unordered_map<std::string, int> encoder_;
    std::vector<std::string> decoder_;
};

class BertTokenizer : public Tiktoken {
public:
    BertTokenizer() = default;
protected:
    virtual void encode(const std::string& str, std::vector<int>& ids) override;
private:
    std::vector<int> word_piece(const std::string& token);
};

class HuggingfaceTokenizer : public Tokenizer {
struct hash_pair_wstring {
    size_t operator()(const std::pair<std::wstring, std::wstring>& p) const {
        auto hash1 = std::hash<std::wstring>{}(p.first);
        auto hash2 = std::hash<std::wstring>{}(p.second);
        // If hash1 == hash2, their XOR is zero.
        return (hash1 != hash2) ? hash1 ^ hash2 : hash1;
    }
};
using BPERanks = std::unordered_map<std::pair<std::wstring, std::wstring>, int, hash_pair_wstring>;
public:
    HuggingfaceTokenizer() = default;
    virtual std::string decode(int id) override;
protected:
    virtual bool load_vocab(std::ifstream& file) override;
    virtual void encode(const std::string& str, std::vector<int>& ids) override;
private:
    void bpe(const std::wstring& token, const BPERanks& bpe_ranks, std::vector<std::wstring>* result);
    BPERanks bpe_ranks_;
    std::unordered_map<uint8_t, wchar_t> b2u_;
    std::unordered_map<wchar_t, uint8_t> u2b_;
    std::unordered_map<std::string, int> encoder_;
    std::vector<std::string> decoder_;
};

#endif // TOKENIZER_hpp