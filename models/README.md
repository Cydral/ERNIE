### Models Directory

The `ERNIE/models/` directory contains trained model files that can be used for various tasks. Currently, it includes:

- **`llm_shakespeare_model.dat`**: 
  This model file allows ERNIE to generate text in the style of William Shakespeare. 

  > 🎭 "To be, or not to be, that is the question..." - ERNIE, channeling Shakespeare

  You can use this model to:
  - Generate Shakespearean-style verses
  - Complete famous Shakespeare quotes
  - Potentially create new sonnets or monologues, with more extensive training:
  > 🔬 While the current model can generate short Shakespearean-like phrases, creating full sonnets or monologues would require more comprehensive and intensive training. This represents an exciting future direction for the project!

  To use the Shakespeare model:

  ```cpp
  // Load the Shakespeare model
  mh_llm_net net;
  deserialize("models/llm_shakespeare_model.dat") >> net;

  // Generate text (example usage)
  string prompt = "To be, or not to be,";
  string generated_text = generate_text(net, prompt);
  cout << "ERNIE's Shakespeare impression:\n" << generated_text << endl;
