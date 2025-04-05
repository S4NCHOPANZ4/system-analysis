# 🔍 Analysis Phases

📄 **[Read Full Report](./Workshop-I/Workshop_I_Report.pdf)**

The analysis conducted for this report was structured into four key phases, each with a specific goal:

---

##  Phases 

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