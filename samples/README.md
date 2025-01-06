# Dlib Transformer Example: Very Small Language Model (VSLM)

This repository contains a minimal example of a **Very Small Language Model (VSLM)** using Dlib's deep learning tools. The example demonstrates how to train and generate text using a small Transformer-based model. The code is fully functional and can be used with the latest `master` branch of the Dlib library.

## Overview

The program provides two main modes:

1. **Train Mode**: Trains a small Transformer-based language model on a character-level corpus extracted from the provided `slm_data.h` file (containing Shakespeare text).
2. **Generate Mode**: Generates new text from a trained model using an initial prompt (also provided in `slm_data.h`).

The model is intentionally kept small to ensure simplicity and efficiency, making it a great educational tool for understanding the mechanics of Transformer models.

## Files

- **`slm_defs.h`**: Contains the Transformer model definitions, including the configuration, network type, and other necessary components.
- **`slm_data.h`**: Contains the training text (`shakespeare_text`) and the initial prompt (`shakespeare_prompt`) for text generation.
- **`slm_basic_train_ex.cpp`**: The main program that implements the training and text generation logic.
