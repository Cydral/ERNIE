## Project Structure

The source code for ERNIE is organized in the ERNIE/sources/ directory, with tokenizer models stored in the ERNIE/tokenizing/ directory. This structure keeps the codebase clean and easy to navigate.

📂 ERNIE<br>
┣ 📂 <b>sources</b><br>
┃ ┣ 📜 vslm.cpp<br>
┃ ┣ 📜 advanced_tokenizer.cpp<br>
┃ ┣ 📜 advanced_tokenizer.hpp<br>
┃ ┣ 📜 cuda_dlib_ext.cu<br>
┃ ┣ 📜 cuda_dlib_ext.cuh<br>
┃ ┣ 📜 dlib_ext.h<br>
┃ ┣ 📜 llm_defs.h<br>
┃ ┗ 📜 data.h<br>
┣ 📂 <b>tokenizing</b><br>
┃ ┣ 📂 en-fr<br>
┃ ┃ ┣ 📜 *.model<br>
┃ ┃ ┗ 📜 *.vocab<br>
┃ ┗ 📂 eu<br>
┃ ┃ ┣ 📜 *.model<br>
┃ ┃ ┗ 📜 *.vocab<br>
┣ 📂 models<br>
┃ ┣ 📜 *.dat<br>
┣ 📂 tests<br>
┃ ┣ 📜 *.txt<br>
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
 
- **`cuda_dlib_ext.cuh`**: 
  - CUDA implementations of custom Dlib extensions
  - Optimized for GPU execution

- **`dlib_ext.h`**: 
  - Declarations of new Dlib processing layers
  - Includes code for CPU host execution

- **`llm_defs.h`**: 
  - Contains the versioned construction of the global generative AI model

- **`data_fr.h`**: 
  - Contains general text examples in French
  - Used for training a language model from scratch
  - Serves as a sample dataset for initial model development and testing

### Notes:

1. The main program (`vslm.cpp`) currently includes its own implementation for creating and using a SentencePiece tokenizer, independent of `tokenizer.hpp`.
2. The custom Dlib extensions are split into CUDA implementations (`cuda_dlib_ext.cu` and `cuda_dlib_ext.cuh`) for GPU execution and declarations (`dlib_ext.h`) for CPU host execution.
3. The global generative AI model construction is versioned and defined in `llm_defs.h`.
4. `data_fr.h` provides a readily available dataset for initial experiments and demonstrations of the model's capabilities with French language text.
5. The pre-calculated UNIGRAM tokenizer models offer ready-to-use tokenization for different languages (currently `en-fr`and `eu`) and scenarios, enhancing the versatility of the ERNIE project.

To build the project, ensure all source files are included in your build configuration. Set `vslm.cpp` as the main entry point for compilation. When using the pre-calculated tokenizer models, make sure to specify the correct model file in your configuration or code.
