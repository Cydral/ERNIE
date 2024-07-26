## Project Structure

The source code for ERNIE is organized in the `ERNIE/sources/` directory.

### Key Files:

- **`vslm.cpp`**: 
  - Main program file and the core of the ERNIE implementation
  - Contains custom layer extensions for the Dlib library
  - Implements the VSLM architecture and training logic
  - Includes utility functions and the program's entry point

- **`tokenizer.hpp`**: 
  - Header file containing "readers" for various tokenizer types
  - Currently not directly used by the main program
  - Provides flexibility for future tokenization method implementations

- **`data_fr.h`**: 
  - Contains general text examples in French
  - Used for training a language model from scratch
  - Serves as a sample dataset for initial model development and testing

### Notes:

1. The main program (`vslm.cpp`) currently includes its own implementation for creating and using a SentencePiece tokenizer, independent of `tokenizer.hpp`.

2. The compact structure with a single main source file (`vslm.cpp`) allows for straightforward management and compilation of the project.

3. `data_fr.h` provides a readily available dataset for initial experiments and demonstrations of the model's capabilities with French language text.

To build the project, ensure all source files are included in your build configuration. Set `vslm.cpp` as the main entry point for compilation.
