# Model Improvement and Safety Answers

### 1. If you only had 200 labeled replies, how would you improve the model without collecting thousands more?

To improve the model with only 200 labels, I would use data augmentation techniques like back-translation or synonym replacement to artificially expand the dataset. I would also leverage transfer learning by fine-tuning a larger, more powerful pre-trained transformer model (e.g., RoBERTa), which can achieve high performance with less task-specific data.

### 2. How would you ensure your reply classifier doesn’t produce biased or unsafe outputs in production?

I would implement a multi-layered safety approach by first ensuring the training data is carefully audited and cleaned of biased or toxic content. In production, I would add a post-processing step where a separate, dedicated safety model (like a moderation API) reviews the classifier's output to flag or filter anything inappropriate before it is used.

### 3. Suppose you want to generate personalized cold email openers using an LLM. What prompt design strategies would you use to keep outputs relevant and non-generic?

I would use a few-shot prompting strategy that includes high-quality examples of personalized openers and clear instructions on tone and length. The prompt would be dynamically populated with specific details about the recipient (e.g., from their LinkedIn profile, recent articles, or company news) to ensure the generated output is highly contextual and unique.
