# System analysis Workshops
<small>Juan David Buitrago Rodriguez - 20242020194</small>
<br>
<small>David Giovanni Aza Carvajal - 20241020137</small>

- [Workshop I](#workshop-i)
- [Workshop II](#workshop-ii)

---

# Workshop II 

📄 **[Read Full Report](./Workshop_2_Design/Workshop_II.pdf)**

Following the initial analysis phase, this workshop focuses on the implementation, evaluation, and iterative refinement of a deep learning model for histological image segmentation.

### 🧠 System Analysis Techniques

In this project, we used **system analysis** to better understand the problem and design the solution step by step. These are some of the techniques we applied:


- **Component Mapping:** We broke the project down into parts like: input data, preprocessing, model, output masks, and evaluation metrics.
- **Pipeline Visualization:** We created flowcharts and diagrams to see how the data moves through the model (from raw image to prediction).
- **Reference Analysis:** We studied successful public Kaggle notebooks and a key scientific paper that helped us understand how CNNs work in histological image segmentation.

---

### 🧬 CNN Implementation/Analysis Process

To build our Convolutional Neural Network (CNN) for classifying histological kidney images, we followed a structured process:

1. **Studied the Domain:** We analyzed histology slide structure and annotation formats, focusing on `.tif` images containing kidney tissue regions.

2. **Reviewed Scientific Literature:** We consulted key sources such as [Deep Learning for Semantic Segmentation in Histology](https://www.mdpi.com/2078-2489/16/3/195), which helped us understand histological image processing and neural network architectures relevant to biomedical tasks. Although the paper focuses on segmentation, we adapted its insights for a classification task.

3. **Implemented the Model:** We developed a custom classifier based on **ResNet50** augmented with **CBAM (Convolutional Block Attention Module)**. This model was pretrained on ImageNet and fine-tuned to distinguish between three classes: *Glomerulus*, *Blood Vessel*, and *Unsure*. The CBAM module improves the model’s attention to relevant spatial and channel features in histological tiles.

4. **Prepared the Data:** Whole-slide `.tif` images were divided into 512×512 tiles. Each tile was preprocessed using resizing and normalization, then passed through the model for classification.

The model workflow is as follows:

- A `.tif` image is split into fixed-size tiles.
- Each tile is preprocessed and passed into the ResNet50_CBAM model.
- The model predicts class probabilities using a softmax layer.
- Predictions per tile are printed and optionally stored in a table for analysis.

This pipeline provides a fast and scalable method for classifying high-resolution histology data using deep learning and attention-enhanced CNNs.


📘 **[Return to Report](./Workshop-II/Workshop_II_Report.pdf)**


---



# Workshop I 


## 🔍 Analysis 

📄 **[Read Full Report](./Workshop-I/Workshop_I_Report.pdf)**

The analysis conducted for this report was structured into four key phases, each with a specific goal:

---

###  Phases 

- **1. Data & Overview Analysis**
  - 📚 **Objective:** Understand the dataset and initial data structure.
  -  **Actions:**
    - Explored the `.tif` image files and metadata.
    - Reviewed competition goals and objectives.
    - The dataset for this competition is located in `/Workshop-I/data/`, exepting the folder `/test` and `/train` folders  that contain the `.tif` images  due to their size (>4GB). These files can be downloaded directly from [Kaggle’s competition page](https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/data).

- **2. Thematic Research**
  - 📚 **Objective:** Investigate domain-specific knowledge.
  -  **Topics Covered:**
    - Histological tissues and blood vessel anatomy.
    - WSI (Whole Slide Imaging) concepts.
    - Role of masks and annotations in biomedical imaging.
    - How Kaggle competitions are typically structured.

- **3. Competition Notebooks Lookup**
  - 📚 **Objective:** Gain insights from public solutions.
  -  **Activities:**
    - Reviewed notebooks on Kaggle.
    - Analyzed preprocessing and visualization techniques.
    - Observed modeling strategies for mask handling.
    - A thorough review of public notebooks related to the competition was conducted, we made particular focus on the [notebook by Ahmed Maher El-Saeidy](https://www.kaggle.com/code/ahmedmaherelsaeidy/hubmap-hacking-the-human-vasculature-dataset). The code from this notebook was used as a reference and implemented under the purpose of understanding the dataflow and how the elements interacted among themselves. This implementation can be found in the folder `/Workshop-I/code`.

- **4. System Comprehension & Analysis**  
   After gathering all the information from the previous phases, we structured the system analysis by mapping out how components interact within the pipeline, summarizing key insights, and formulating visual overviews. All these findings were integrated into our final **[report](./Workshop-I/Workshop_I_Report.pdf)**.

---

📘 **[Return to Report](./Workshop-I/Workshop_I_Report.pdf)**


