# Deepfake Detection with RL-Assisted Active Learning and Explainable Predictions

## Problem Statement

Training deepfake detection systems requires massive amounts of labeled data. However, labeling images is slow and expensive. Standard systems label images randomly, wasting money on data the model already understands. Additionally, most detectors are "black boxes," offering no visual explanation for their predictions, which erodes user trust.

## Our Solution

We propose a resource-efficient and transparent detection pipeline that learns how to learn more effectively:

- **CNN Classifier**: Detects Real vs. Fake images (Target: 85%+ Accuracy)
- **RL-Based Active Learning**: Intelligently selects which unlabeled images to label next using Q-learning. Instead of random guessing, the agent learns optimal sampling strategies to minimize labeling costs
- **GradCAM Explainability**: Generates visual heatmaps showing which facial regions influenced predictions, building user trust through transparency

## Technical Workflow (The Learning Loop)

### Phase 1: The Cold Start (Initial Knowledge) 🏁

1. Start with a small "Seed Set" of pre-labeled data (500 images: 250 real, 250 fake)
2. Train baseline CNN on this initial set → achieves ~70-72% accuracy
3. The model is now ready to learn from additional data

### Phase 2: The Active Learning Loop (Smart Data Selection) 🔄

This loop repeats to make the model smarter without labeling everything:

1. **Scan**: CNN evaluates a pool of Unlabeled Images, outputting confidence scores for each
2. **Select (RL Agent)**: The RL Agent observes the state (confidence, accuracy, budget) and uses a Q-Table to choose one of 4 strategies:
   - **Strategy A**: Uncertainty Sampling (Pick low confidence)
   - **Strategy B**: High Confidence (Check for errors)
   - **Strategy C**: Random (Explore)
   - **Strategy D**: Class Balance (Equalize Real/Fake)
3. **Label**: The selected image is labeled (simulated by revealing hidden ground truth)
4. **Retrain**: CNN fine-tunes on the expanded training set
5. **Reward**: The Agent receives a reward if accuracy improves, learning which strategy is best

### Phase 3: Deployment (User-Facing Product) 🚀

1. User uploads image
2. System outputs:
   - Verdict ("FAKE - 92% confidence")
   - Evidence (GradCAM heatmap highlighting suspicious facial regions)

## Expected Results

- **Efficiency**: Achieve 85% accuracy using 40% fewer labeled images than random sampling
- **Transparency**: All predictions include visual heatmap explanations

## Dataset & Tools

- **Data**: FaceForensics++ (Subset of ~2,000 images for rapid iteration)
- **Stack**: Python, TensorFlow/Keras, NumPy (Q-table), OpenCV, Streamlit

