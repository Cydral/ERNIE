# ERNIE
<p><i>This README provides a comprehensive overview of the ERNIE project, including its background, key features, installation instructions, usage examples, and acknowledgments. It highlights the custom layers developed for the project and emphasizes its role in extending Dlib's capabilities for NLP tasks. The content is structured to be informative and engaging for potential users and contributors on GitHub.</i></p>
<p align="center"><img src="https://github.com/Cydral/ERNIE/blob/main/ERNIE_logo.png" alt=""></p>

## Introduction
Large Language Models (LLMs) have revolutionized natural language processing, demonstrating remarkable capabilities in understanding and generating human-like text. However, they come with significant challenges:

- **Computational Demands**: Traditional LLMs require substantial computational resources, often necessitating powerful GPUs and high-performance hardware.
- **Memory Consumption**: These models consume large amounts of RAM, limiting their deployment on resource-constrained devices.
- **Training Stability**: Achieving stable training for deep neural networks, especially in language tasks, remains a complex challenge.
- **Energy Efficiency**: The power consumption of large models raises concerns about their environmental impact and operational costs.

ERNIE (Efficient Rapid Neural Intelligence Engine) addresses these challenges by implementing a Very Small Language Model (VSLM) using the Dlib C++ Library. This project showcases how to extend Dlib's capabilities to handle advanced NLP tasks, focusing on transformer-based architectures while maintaining efficiency and scalability.

## Project Overview

ERNIE is an ongoing development project that demonstrates the implementation of a compact yet powerful language model using Dlib. Key aspects of the project include:

- **Environment**: Primarily designed for Microsoft Windows (console mode), compilable with Microsoft Visual Studio 2022 64-bit (version 17.8.5 used in development).
- **Hardware Utilization**: Supports both GPU (CUDA) and CPU operations, with compile-time options for flexibility.
- **Cross-Platform Potential**: With minimal adaptations, the codebase can be ported to Linux environments.

> **Note**: This project is in active development and stabilization. While initial results are promising, users should expect ongoing improvements and potential changes.

## Project Structure

The source code for ERNIE is organized in the `ERNIE/sources/` directory. This structure keeps the codebase clean and easy to navigate.

📂 ERNIE<br>
 ┣ 📂 sources<br>
 ┃ ┣ 📜 vslm.cpp<br>
 ┃ ┣ 📜 tokenizer.hpp<br>
 ┃ ┣ 📜 data_fr.h<br> 
 ┃ ┣ 📜 *.model<br>
 ┃ ┗ 📜 *.vocab<br>
 ┣ 📂 models<br>
 ┃ ┣ 📜 *.dat<br>
 ┣ 📜 README.md<br>
 ┗ 📜 LICENSE<br>

### Key Files:

- **`vslm.cpp`**: 
  - Main program file and the core of the ERNIE implementation
  - Contains custom layer extensions for the Dlib library
  - Implements the VSLM (Very Small Language Model) architecture and training logic
  - Includes utility functions and the program's entry point

- **`tokenizer.hpp`**: 
  - Header file containing "readers" for various tokenizer types
  - Currently not directly used by the main program
  - Provides flexibility for future tokenization method implementations

- **`data_fr.h`**: 
  - Contains general text examples in French
  - Used for training a language model from scratch
  - Serves as a sample dataset for initial model development and testing

- **Pre-calculated UNIGRAM Tokenizer Models**:
  - `.model` and `.vocab` files for UNIGRAM tokenizers
  - Includes models for: 3k, 8k, 20k and 40k vocabulary

### Notes:

1. The main program (`vslm.cpp`) currently includes its own implementation for creating and using a SentencePiece tokenizer, independent of `tokenizer.hpp`.
2. The compact structure with a single main source file (`vslm.cpp`) allows for straightforward management and compilation of the project.
3. `data_fr.h` provides a readily available dataset for initial experiments and demonstrations of the model's capabilities with French language text.
4. The pre-calculated UNIGRAM tokenizer models offer ready-to-use tokenization for different languages (currently EN-FR) and scenarios, enhancing the versatility of the ERNIE project.

To build the project, ensure all source files are included in your build configuration. Set `vslm.cpp` as the main entry point for compilation. When using the pre-calculated tokenizer models, make sure to specify the correct model file in your configuration or code.

## Key Features

1. **Custom Matrix-Based Processing Layers**: Optimized for Dlib's tensor structure, enhancing performance in language tasks.
2. **Specialized LLM Input Layers**: Including embedding injection and positional encoding, crucial for transformer architectures.
3. **Comprehensive Training Pipeline**: Complete example of training a language model from scratch.
4. **Text Generation Capabilities**: Showcases the model's ability to generate coherent text based on learned patterns.
5. **Benchmarking Suite**: Includes various tests, notably the "Shakespeare test," demonstrating the model's text generation prowess.
6. **Extensibility**: Serves as a template for further extending Dlib's capabilities in NLP tasks.

## New Custom Layers

ERNIE introduces several custom layers to Dlib, showcasing how to extend the library's functionality for specific NLP tasks. These layers are designed with a focus on matrix-based operations, optimizing for Dlib's tensor structure.

### 1. Embedding Layer (`embedding_`)
```cpp
template<int num_embeddings_, int embedding_dim_>
class embedding_ {
    // ... (code details)
};
```
This layer implements word embeddings, a crucial component in NLP models. It transforms input tokens into dense vector representations.
#### Key Features:
- Customizable embedding dimensions and vocabulary size
- Efficient lookup and update mechanisms
- Support for learning rate multipliers

### 2. Positional Encoding Layer (positional_encoding_)
```cpp
template<int sequence_dim_, int embedding_dim_>
class positional_encoding_ {
    // ... (code details)
};
```
Implements positional encoding, allowing the model to understand the order of tokens in a sequence.
#### Key Features:
- Sinusoidal positional encoding
- Customizable sequence length and embedding dimension
- Efficient forward and backward pass implementations

### 3. Linear Layer (linear_)
```cpp
template <unsigned long num_outputs_, linear_bias_mode bias_mode_>
class linear_ {
    // ... (code details)
};
A custom linear (fully connected) layer with optional bias.
```
#### Key Features:
- Configurable output size and bias mode
- Efficient matrix multiplication using Dlib's BLAS interface
- Support for learning rate multipliers

### 4. Masked Attention Layer (masked_attention_)
```cpp
class masked_attention_ {
    // ... (code details)
};
```
Implements masked self-attention, a core component of transformer models.
#### Key Features:
- Efficient masking mechanism
- Support for both training and inference modes
- Optimized for Dlib's tensor operations

### 5. Softmax Layer (softmaxm_)
```cpp
class softmaxm_ {
    // ... (code details)
};
```
A custom softmax implementation optimized for matrix-based operations.
#### Key Features:
- Efficient computation of softmax across matrix rows
- Handles special cases like all-negative infinity inputs
- Backward pass implementation for gradient computation

### 6. Scale Weights Layer (scale_weights_)
```cpp
class scale_weights_ : public multiply_ {
    // ... (code details)
};
```
A utility layer for scaling weights in attention mechanisms.
#### Key Features:
- Automatic scaling based on embedding size and number of attention heads
- Inherits from Dlib's multiply_ layer for efficiency

### 7. Transformer Block
```cpp
template <typename SUBNET>
using transformer_block = feed_forward_linear<embedding_size,
    multihead_attention_block<SUBNET>>;
```
Combines multiple custom layers to create a complete transformer block.
#### Key Features:
- Implements the standard transformer architecture
- Combines multi-head attention with feed-forward networks
- Utilizes layer normalization for stability

These custom layers demonstrate how to extend Dlib's functionality for specific NLP tasks while maintaining compatibility with the library's existing infrastructure. They serve as examples for further customization and optimization in language processing applications.

## Installation
### Prerequisites
- Microsoft Visual Studio 2022 (64-bit)
- CUDA Toolkit (for GPU support)
- Dlib C++ Library
- Boost C++ Libraries
- SentencePiece Library

## Examples
Shakespeare Test
The Shakespeare test demonstrates ERNIE's ability to learn and generate text in the style of William Shakespeare. Here's a sample output:

```mizar
Input: "To be, or not to be, that is the"

Generated text: "To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;"
```
This example showcases how ERNIE can capture the essence of Shakespeare's writing style, including vocabulary, meter, and thematic elements.

## Acknowledgements
This project would not have been possible without the incredible work of the Dlib community.
ERNIE stands on the shoulders of giants in the field of machine learning and natural language processing. We are grateful for the wealth of knowledge and tools provided by the community.
> ERNIE is an ongoing project, and we're excited to see how it evolves with community input and advancements in NLP research. Stay tuned for updates, and happy coding!
