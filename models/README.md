### Models Directory

The `ERNIE/models/` directory contains trained model files that can be used for various tasks. Currently, it includes:

- **`llm_shakespeare_model_a.dat`**: 
  This model file demonstrates ERNIE's ability to memorize and reproduce text from William Shakespeare's works.

  > 🎭 "To be, or not to be, that is the question..." - ERNIE, reciting Shakespeare

  You can use this model to:
  - Reproduce famous Shakespeare quotes
  - Demonstrate the model's memorization capabilities
  
  > ⚠️ **Note:** This initial model primarily showcases ERNIE's capacity for memorization rather than true creative generation. It essentially recites learned passages when given familiar prompts.

```mizar
Input: "To be or not to be—that is the ques"

Generated text: "To be or not to be—that is the question:
Whether ’tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And, by opposing, end them. To die, to sleep—
No more—and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to—’tis a consummation
Devoutly to be wished. To die, to sleep—
To sleep, perchance to dream. Ay, there (...)"
```

  🔍 Sequence Length Impact:
It's worth noting that ERNIE's ability to accurately recite the learned text depends significantly on the input sequence length. For instance, in Hamlet's soliloquy, the phrase "To die, to sleep—" appears twice with different continuations. ERNIE correctly disambiguates between these occurrences due to its sufficiently large context window. This showcases how the model uses the broader context to accurately predict the next characters, demonstrating not just memorization, but also the capacity to utilize contextual information for precise text reproduction.

- **`llm_shakespeare_model_b.dat`**: 
  This advanced model represents ERNIE's attempt to generate text in the style of Shakespeare after training on his complete works.

  > 🖋️ "Shall I compare thee to a winter's night?" - ERNIE, attempting Shakespeare-style generation

  You can use this model to:
  - Generate Shakespearean-style verses
  - Attempt to create new sonnets or monologues
  - Explore ERNIE's understanding of Shakespeare's writing style

  > 🔬 While this model aims to capture Shakespeare's style, its output may vary in quality and authenticity. It represents an exciting step towards more sophisticated text generation!

📊 Performance Metrics:
  - Model A (Memorization): Typically achieves near-perfect reproduction of learned passages.
  - Model B (Generation): Performance varies. Check the 'relevance score' in the test output for an indication of how well it captures Shakespeare's style and vocabulary.

We continue to refine these models to improve ERNIE's language understanding and generation capabilities.
