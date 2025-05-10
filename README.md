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

### 🧬 CNN Creation Process

To build our Convolutional Neural Network (CNN) for blood vessel segmentation, we followed these steps:

1. **Studied the Domain:** We explored the structure of histological slides and how vessels are annotated in `.tif` masks.
2. **Read Scientific Literature:** A key paper, [Deep Learning for Semantic Segmentation in Histology](https://www.mdpi.com/2078-2489/16/3/195), helped us understand the best practices. It showed how CNNs are used for similar tasks and explained common challenges and architectures (like U-Net).
3. **Designed the Model:** Inspired by the paper and notebooks, we implemented a U-Net-based model with a backbone pretrained on ImageNet.
4. **Prepared the Data:** We used data augmentation, normalization, and converted `.tif` images into smaller patches suitable for training.
The model works in the following way:

- Each tile is processed by a neural network (ResNet50 with attention blocks) to extract important features.
- These features are passed through layers that help the model focus on patterns, reduce noise, and improve accuracy.
- Finally, a classifier assigns a label to each tile — such as *blood vessel*, *glomerulus*, or *unsure*.

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


