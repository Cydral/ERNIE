// Copyright (C) 2024  Cydral
// License: MIT Software License. See LICENSE for the full license.
// GitHub: https://github.com/Cydral/ERNIE

#ifndef ADVANCED_TOKENIZER_hpp
#define ADVANCED_TOKENIZER_hpp

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

// ----------------------------------------------------------------------------------------

class advanced_tokenizer
{
    /*!
        WHAT THIS OBJECT REPRESENTS
            This object represents a base class for advanced tokenizers that support
            modern NLP techniques such as SentencePiece, TikToken, BERT, and HuggingFace
            tokenization methods.

        THREAD SAFETY
            It is safe to access const methods of this object from multiple threads.
            However, any operation that modifies the object is not thread-safe.
    !*/

public:
    static constexpr int MAGIC_NUMBER = 430;

    enum tokenizer_type
    {
        SENTENCEPIECE = 0,
        TIKTOKEN = 1,
        BERT = 2,
        HUGGINGFACE = 3
    };

    advanced_tokenizer() = default;
    virtual ~advanced_tokenizer() = default;

    static advanced_tokenizer* create_tokenizer(const std::string& filename);
    /*!
        ensures
            - Creates and returns a tokenizer based on the contents of the given file.
        throws
            - std::runtime_error if the file cannot be read or is invalid.
    !*/

    bool is_stop(int token) const;
    /*!
        ensures
            - Returns true if the given token is a stop token, false otherwise.
    !*/

    bool is_special(int token) const;
    /*!
        ensures
            - Returns true if the given token is a special token, false otherwise.
    !*/

    std::vector<int> encode(const std::string& str);
    /*!
        ensures
            - Encodes the given string into a vector of token IDs.
        throws
            - std::runtime_error if the string cannot be encoded.
    !*/

    virtual std::string decode(int id) = 0;
    /*!
        ensures
            - Decodes the given token ID into a string.
        throws
            - std::runtime_error if the ID cannot be decoded.
    !*/

protected:
    virtual void load_special(std::ifstream& file);
    virtual bool load_vocab(std::ifstream& file) = 0;
    virtual void encode(const std::string& str, std::vector<int>& ids) = 0;

    std::vector<int> special_tokens_;
    std::vector<int> stop_tokens_;
    std::vector<int> prefix_tokens_;
};

// ----------------------------------------------------------------------------------------

class sentencepiece_tokenizer : public advanced_tokenizer
{
    /*!
        WHAT THIS OBJECT REPRESENTS
            This object represents a SentencePiece tokenizer.

        THREAD SAFETY
            It is safe to access const methods of this object from multiple threads.
            However, any operation that modifies the object is not thread-safe.
    !*/

public:
    sentencepiece_tokenizer() = default;

    std::string decode(int id) override;
    /*!
        ensures
            - Decodes the given token ID into a string using SentencePiece method.
        throws
            - std::runtime_error if the ID cannot be decoded.
    !*/

protected:
    bool load_vocab(std::ifstream& file) override;
    void encode(const std::string& str, std::vector<int>& ids) override;

    enum model_type {
        UNIGRAM = 1,
        BPE = 2,
        WORD = 3,
        CHAR = 4
    };
    enum piece_type {
        NORMAL = 1,
        UNKNOWN = 2,
        CONTROL = 3,
        USER_DEFINED = 4,
        UNUSED = 5,
        BYTE = 6
    };
    struct sentence_piece {
        std::string piece;
        float score;
        piece_type type = piece_type::NORMAL;
        sentence_piece() {}
        sentence_piece(const std::string& p, float s, piece_type t) : piece(p), score(s), type(t) {}
    };
    using encode_result = std::vector<std::pair<string_view_, int>>;

private:
    // Model train type
    model_type type_ = BPE;
    // Byte fall back enable
    bool byte_fall_back_ = true;
    // Unknown id
    int unk_id_ = 0;

    // Pieces from model
    std::vector<sentence_piece> sentence_pieces_;
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
    encode_result bpe_encode(string_view_ normalized, float alpha = 0.f);
};

// ----------------------------------------------------------------------------------------

class tiktoken_tokenizer : public advanced_tokenizer
{
    /*!
        WHAT THIS OBJECT REPRESENTS
            This object represents a TikToken tokenizer.

        THREAD SAFETY
            It is safe to access const methods of this object from multiple threads.
            However, any operation that modifies the object is not thread-safe.
    !*/

public:
    tiktoken_tokenizer() = default;

    std::string decode(int id) override;
    /*!
        ensures
            - Decodes the given token ID into a string using TikToken method.
        throws
            - std::runtime_error if the ID cannot be decoded.
    !*/

protected:
    virtual bool load_vocab(std::ifstream& file) override;
    virtual void encode(const std::string& str, std::vector<int>& ids) override;

    std::unordered_map<std::string, int> encoder_;
    std::vector<std::string> decoder_;
};

// ----------------------------------------------------------------------------------------

class bert_tokenizer : public tiktoken_tokenizer
{
    /*!
        WHAT THIS OBJECT REPRESENTS
            This object represents a BERT tokenizer.

        THREAD SAFETY
            It is safe to access const methods of this object from multiple threads.
            However, any operation that modifies the object is not thread-safe.
    !*/

public:
    bert_tokenizer() = default;

protected:
    void encode(const std::string& str, std::vector<int>& ids) override;

private:
    std::vector<int> word_piece(const std::string& token);
};

// ----------------------------------------------------------------------------------------

class huggingface_tokenizer : public advanced_tokenizer
{
    /*!
        WHAT THIS OBJECT REPRESENTS
            This object represents a HuggingFace tokenizer.

        THREAD SAFETY
            It is safe to access const methods of this object from multiple threads.
            However, any operation that modifies the object is not thread-safe.
    !*/

    struct hash_pair_wstring {
        size_t operator()(const std::pair<std::wstring, std::wstring>& p) const {
            auto hash1 = std::hash<std::wstring>{}(p.first);
            auto hash2 = std::hash<std::wstring>{}(p.second);
            // If hash1 == hash2, their XOR is zero
            return (hash1 != hash2) ? hash1 ^ hash2 : hash1;
        }
    };
    using bpe_ranks = std::unordered_map<std::pair<std::wstring, std::wstring>, int, hash_pair_wstring>;

public:
    huggingface_tokenizer() = default;

    std::string decode(int id) override;
    /*!
        ensures
            - Decodes the given token ID into a string using HuggingFace method.
        throws
            - std::runtime_error if the ID cannot be decoded.
    !*/

protected:
    bool load_vocab(std::ifstream& file) override;
    void encode(const std::string& str, std::vector<int>& ids) override;

private:
    bpe_ranks bpe_ranks_;
    std::unordered_map<uint8_t, wchar_t> b2u_;
    std::unordered_map<wchar_t, uint8_t> u2b_;
    std::unordered_map<std::string, int> encoder_;
    std::vector<std::string> decoder_;

    void bpe(const std::wstring& token, const bpe_ranks& bpe_ranks, std::vector<std::wstring>& result);
};

#endif // ADVANCED_TOKENIZER_hpp