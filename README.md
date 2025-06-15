Project Title: Pneumonia Detection using Deep Learning on Chest X-Ray Images

🩺 What is Pneumonia? Pneumonia is an inflammatory condition of the lungs, primarily affecting the alveoli (small air sacs). Symptoms typically include coughing (productive or dry), fever, chest pain, and difficulty breathing. It can vary in severity and is most commonly caused by bacterial or viral infections. Risk factors include chronic respiratory diseases (e.g., asthma, COPD), diabetes, smoking history, heart failure, immune suppression, and more.

📂 Dataset Description: Source: Chest X-ray (anterior-posterior) images from pediatric patients.

Structure: 3 folders: train, test, val

Each folder contains 2 categories: PNEUMONIA and NORMAL

Total Images: 5,863 JPEG files

Preprocessing: Images converted to grayscale Resized to 150×150 pixels Labeled based on folder path

⚙️ Technical Stack & Libraries Used: Data Handling & Processing: NumPy, Pandas, OpenCV Visualization: Matplotlib, Seaborn Deep Learning Framework: Keras (with TensorFlow backend) Model Evaluation: sklearn.metrics

🔄 Data Preprocessing & Augmentation: Normalization: Pixel values scaled to [0, 1]

Data Augmentation (via ImageDataGenerator): Random rotation (up to 30°) Zoom range (20%) Horizontal & vertical shifts (10%) Horizontal flipping

🧠 Model Architecture (CNN): Layer Type Filters Activation Additional Notes Input Conv2D 32 ReLU BatchNorm + MaxPooling Hidden 1 Conv2D 64 ReLU Dropout + BatchNorm + MaxPooling Hidden 2 Conv2D 64 ReLU BatchNorm + MaxPooling Hidden 3 Conv2D 128 ReLU Dropout + BatchNorm + MaxPooling Hidden 4 Conv2D 256 ReLU Dropout + BatchNorm + MaxPooling Dense Fully Connected 128 ReLU Dropout Output Dense 1 Sigmoid Binary classification

Compilation:

Optimizer: RMSprop Loss Function: Binary Crossentropy Metric: Accuracy

Learning Rate Scheduling: ReduceLROnPlateau monitors val_accuracy Reduces LR by factor 0.3 if no improvement for 2 epochs

Minimum LR: 1e-6

📈 Training Details: Epochs: 12 Batch Size: 32 Validation Strategy: 20% of training used for validation Fit: Using augmented data (flow() method)

📊 Results & Evaluation: Test Accuracy: Printed post-training Confusion Matrix: Visualized using Seaborn heatmap Metrics Reported: Precision, Recall, F1-Score Class labels: Pneumonia (Class 0), Normal (Class 1)

🔍 Model Insights: Correct Predictions (Examples): Displayed grayscale images where predicted and actual labels matched.

Incorrect Predictions (Examples): Displayed misclassified images with actual vs. predicted classes.

Conclusion: This convolutional neural network (CNN) model effectively classifies chest X-ray images into Pneumonia and Normal classes. Through data augmentation, normalization, and learning rate tuning, the model achieves good generalization and avoids overfitting. Visual analysis of both correct and incorrect predictions provides additional insights into model performance.
