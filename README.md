# Students' Final Projects  for 2025 Fall UVa CS -ML-Undergraduate

---

## Related: Students' projects codebase from this course's past offerings in 2020 and 2019
- [machine-learning-uva2019-students-deep2reproduce/](https://github.com/Qdata4Capstone/machine-learning-uva19-students-deep2reproduce/)
- [machine-learning-uva2020-students-ml for good use](https://github.com/Qdata4Capstone/machine-learning20f-learning4good-projects)

## The course website: [https://qiyanjun.github.io/2025Fall-UVA-CS-MachineLearningDeep/](https://qiyanjun.github.io/2025Fall-UVA-CS-MachineLearningDeep/)

---

### Index of the students' team projects:  

As a class, Students built everything from AI systems for medical image analysis and financial fraud detection to local LLM-powered assistants, trash-sorting computer vision models, DJ song-mixing recommenders, chess rating predictors, and even outfit recommendation apps. 



#### Healthcare & Medical AI

| Index | Keywords | Video | Summary |
|-------|----------|-------|--|
| t03 | ML_gradCam |  | This project utilizes Grad-CAM to analyze and compare deep learning models like ResNet50, DenseNet121, and a Simple CNN for diagnosing lung diseases from chest X-rays, aiming to increase clinical trust by visualizing the specific features the models focus on.
| t06 | DiabetesForecasting | [link](https://youtu.be/4sVgWTkxrcI) | This project explores the use of machine learning models, specifically LSTM and Random Forest, to forecast diabetes progression by analyzing complex patient health datasets to improve predictive accuracy and clinical decision-making. |
| t13 | CSMedicalImage |  | This project trains a ResNet-50 CNN on the RSNA chest X-ray dataset to classify pneumonia vs. normal lungs and uses Grad-CAM heatmaps to make predictions interpretable for clinical decision support. |
| t15 | ML-skinI | [link](https://youtu.be/dWKxdcSMWO4) | This project proposes using EfficientNet-B0 to classify skin lesions from the HAM10000 dataset, achieving an accuracy of 88% with an 82.92% reduction in parameters compared to traditional ResNet50 models. |
| t26 | MLBrain_tumor |  | This project trains a 2D U-Net on multimodal BraTS MRI scans to automatically segment glioblastoma subregions (necrotic core, edema, and enhancing tumor) at the pixel level with very high accuracy, reducing manual effort and bias in brain tumor delineation. |


#### Finance & Risk Analytics

| Index | Keywords | Video | Summary |
|-------|----------|-------|--|
| t10 | MarketMinds-Headlines-to-Returns | [link](https://youtu.be/iN-jLGnTcNQ) | This project explores the use of FinBERT contextual embeddings compared to traditional market momentum indicators for predicting next-day DJIA movements, ultimately rejecting the hypothesis as complex NLP models underperformed simple technical baselines. |
| t11 | FairCreditPredictionML |  | This project develops a production-ready, fairness-aware credit scoring system that uses CDI-based proxy groups, reweighing, group-specific thresholds, and SHAP/LIME explanations with full monitoring infrastructure to make more equitable and transparent credit card approval decisions. |
| t16 | CreditCardFraud | [link](youtube.com/watch?v=O41fVjY96qk&feature=youtu.be) | This project builds and compares a class-imbalance-aware credit card fraud detector using logistic regression (fast baseline with balanced weights) versus a Keras neural network (nonlinear classifier), showing the neural network achieves a far better precision–recall tradeoff on the highly imbalanced European transactions dataset. |
| t20 | FraudulentAccount | [link](https://youtu.be/lVr0j_KLIIk) | This project builds a fraud detection pipeline using a balanced random forest trained on a highly imbalanced 1M-row Base Application Fraud dataset, tuned via precision–recall tradeoffs and SHAP analysis to better flag fraudulent bank account applications while controlling review costs. |
| t24 | StockPrice | [link](https://drive.google.com/file/d/1_knZ4cJ5R_VtIu8JD36ALOirAj8IVv1H/view?usp=sharing) | This research project used the finBERT model to analyze the sentiment of over 13,000 financial headlines and found a very weak correlation (less than 0.02) between news sentiment and daily stock price changes. |
| t27 | ETF-risk | [link](https://youtu.be/KxbXr0T0Rvw) | This project utilizes an interpretable Logistic Regression model and SHAP analysis to provide early-warning signals for dividend instability risk in ETFs, achieving roughly 74% accuracy in predicting whether dividends will fail to grow over a forward-looking 12-month period. |


#### NLP, LLMs & Education Assistants

| Index | Keywords | Video | Summary |
|-------|----------|-------|--|
| t07 | Local-RAG-Vector-Search-System | [link](https://www.youtube.com/watch?v=N8b3yETTTbA) | This project implements a fully local RAG system that indexes user documents with sentence-transformer embeddings and FAISS, then uses an on-device LLaMA model to provide privacy-preserving, semantically grounded document question answering with cited sources. |
| t12 | ML_major-news |  | This project builds a Reddit-based pipeline that scores post-title sentiment with a tuned TF-IDF + Linear SVM, detects "major events" as weeks with unusually high engagement (robust z-scores over comments/upvotes), and compares pre/during/post sentiment shifts across subreddits to surface interpretable, event-centric insights. |
| t17 | CSAIassistant | [link](https://youtu.be/IAYtRCeQ7_s) | This project builds a UVA AI Course Assistant that unifies SIS, HoosList, RateMyProfessor, and TheCourseForum data in a vector-backed Gemini chatbot to provide students with personalized course recommendations, schedule planning, and advisor-style guidance with memory. |
| t18 | MicroProgram |  | This project explores the use of large language models to automatically generate and debug micro-programs for hardware with restricted instruction sets and limited resources, demonstrating that GPT-5 can achieve 100% accuracy in correcting incomplete low-level control sequences when provided with a reference implementation. |
| t19 | canvasGPT | [link](https://www.youtube.com/watch?v=IBh6QZ9l06A) | CanvasGPT is an Electron desktop app that connects to a student's Canvas account to automatically discover and ingest course data (including linked external sites), unify and semantically index the otherwise unstructured content, and deliver intelligent retrieval plus proactive deadline/update alerts—optionally exposed through an MCP LLM interface for homework assistance. |


#### Computer Vision & Image Processing (Non-Medical)

| Index | Keywords | Video | Summary |
|-------|----------|-------|--|
| t02 | ML_YoloTrash | [link1](https://youtu.be/L-Bif0_NOkU),[link2](https://youtu.be/L-Bif0_NOkU) | This project fine-tunes a YOLOv8-seg model on a synthetically-generated dataset derived from TrashNet to perform real-time instance segmentation of recyclable objects, specifically plastic, in complex real-world environments. |
| t05 | Human-In-the-LoopRL-ImageSynth | [link](https://drive.google.com/file/d/1ztj9GcXTr1e4RqEcebv-Fzs70U8VtJij/view?usp=drive_link) | This project builds a human-in-the-loop reinforcement learning system that learns an individual artist's aesthetic preferences from 1–5 ratings (via Q-learning, Deep Q-learning, and PPO) to steer image generation toward their personal style and increase artist autonomy. |
| t14 | InventoryMonitor | [link](https://www.youtube.com/watch?v=TyAtKc9BPjs) | This project builds a camera-based fridge monitoring system using a custom CNN and database-backed web app to track items and expiration dates, then generate recipe suggestions and nutrition macros to reduce household food waste.  
| t21 | ML_Image_Colorization_Presentation | [link](https://myuva-my.sharepoint.com/:p:/g/personal/utw5es_virginia_edu/IQBKualfcb7LSYgeoj0j1Ww0ARdaqcZu8mnCsosx3DE0vdI?e=xObXdm) | This project explores image colorization by using a GAN with a U-Net encoder-decoder architecture to infer realistic color channels from grayscale inputs, finding that while the model achieves low L1 loss, objective metrics like SSIM often misalign with the subjective visual quality of the results. |
| t25 | DrunkDriver |  | This project builds a CNN-based drunk-driving detection pipeline that extracts and crops faces from sober/drunk videos into frames, trains a multi-layer convolutional classifier, and achieves ~95% accuracy as a fast, less-intrusive intoxication screening tool. |
| t30 | Mushroom | [link](https://youtu.be/5bpg49MUFBY) | This project explores the use of CNNs, specifically a baseline CNN and MobileNetV2, to classify mushroom images as edible or poisonous, ultimately achieving an overall test accuracy of 80.67% and a poisonous mushroom recall of approximately 84%. |


#### Sports, Entertainment & Lifestyle

| Index | Keywords | Video | Summary |
|-------|----------|-------|--|
| t04 | ML_soccerplayer | [link](https://youtu.be/kHUYCva1sxY) | This project utilizes a Random Forest machine learning model trained on FBREF soccer statistics to identify suitable player replacements by analyzing key playing-style attributes and historical performance metrics. |
| t09 | ChessElo | [link](https://youtu.be/SgSRFI_cDYs) | This project develops a machine learning regression model that utilizes Stockfish engine evaluations to analyze PGN files and predict chess players' Elo ratings with an average accuracy within 170 points of the actual result. |
| t22 | SA-musicians | [link1](https://youtu.be/75aeonsjKK0),[link2](https://youtu.be/ZwsHOmrfXFE),[link3](https://youtu.be/l7HYH53H9nw),[link4](https://youtu.be/GlcwNCH_gfM),[link5](https://youtu.be/BruqunChwj8) | This project builds a multi-dimensional artist profiling pipeline that combines transformer-based lyric sentiment, Sentence-BERT theme clustering, cross-platform public-perception sentiment with a RAG QA layer, and SARIMAX forecasting of Spotify popularity trends for six top music artists. |
| t28 | DJ_Mixing_Recommendation_Final | [link](https://www.youtube.com/watch?v=X2zmxph66xg&feature=youtu.be) | This project builds a DJ song-transition recommender that uses Spotify audio features plus DJ mixing rules (±6 BPM beatmatching, Camelot key compatibility, and energy flow) and compares a rule-based filter, an audio-similarity baseline, and an XGBoost hybrid model to rank the top 10 most "mixable" next tracks for any given song.  |
| t31 | ML-outfit |  | This project builds Bundle Buddy, a Random Forest–based, feedback-driven system that uses weather API data and user-reported activity and comfort history to personalize daily outfit recommendations with about 87% accuracy. |


#### Industrial Engineering & ML Theory

| Index | Keywords | Video | Summary |
|-------|----------|-------|--|
| t01 | ML_Anomaly | [link](https://youtu.be/lEvAmGlhEuE) | This project explores and compares unsupervised and semi-supervised hybrid LSTM models, such as LSTM-OC-SVM and LSTM-DBSCAN, to detect energy consumption anomalies in buildings using data-driven analytical techniques. |
| t08 | Autodifferentiation | [link](https://youtu.be/1IJK0WWCo3E) | This project explains and implements reverse-mode automatic differentiation by building a small autodiff library, demonstrating it on simple computational graphs, and benchmarking it on a neural network against industry tools. |
| t23 | ML_EngineFailure | [link](https://www.youtube.com/watch?v=jBvGLsg9_vc) | This project uses the NASA C-MAPSS turbofan dataset to train and compare Random Forest and SVM classifiers that flag aircraft engines within five cycles of failure from time-series sensor and operating-condition data with ~99% accuracy, enabling earlier and more interpretable maintenance decisions. |
| t29 | When_Does_ML_Fail_Presentation |  | This project systematically stress-tests logistic regression, random forest, and MLP classifiers on the UCI Adult Income dataset under feature noise, label corruption, and distribution shift to reveal how common ML models fail and why accuracy alone can mask their brittleness. |


### Guide to students: How to PR? 

For those who haven't submitted your project code yet, please follow the instructions below to upload your work to the course repository.

Step 1: Set up your local branch
- Go to the course repository and click Fork: https://github.com/Qdata4Capstone/uva-machine-learning-25f-projects
- Go to your new forked repository and clone it to your local environment:
  - git clone https://github.com/<your-username>/uva-machine-learning-25f-projects.git
- Navigate into the cloned folder and add the original repository as an upstream remote:
  - git remote add upstream https://github.com/Qdata4Capstone/uva-machine-learning-25f-projects.git

Step 2: Prepare your code:
- For each team, please create a folder named `team-XX` corresponding to your team ID (e.g., team-1, team-11, team-111). 
- Inside this folder, include the following:
  - src/: A subfolder containing all source code.
  - data/: A subfolder with the data required to reproduce results.
    - Note: If the data cannot be uploaded, include a markdown file describing how to collect it.
  - `requirements.txt`: A file listing required packages. (Format [reference](https://pip.pypa.io/en/stable/reference/requirements-file-format/))
  - `README.md`: A markdown file describing the folder content. You can view an example [here](https://github.com/QData/TextAttack). Your README should include:
    - Project Title
    - Team ID and Members
    - Overview: A brief introduction to the project.
    - Usage: How to run the code to get core results.
    - (Optional) Setup: Instructions for environment setup (if non-trivial).
    - (Optional) Video: A link to your demo video with a brief description.
- You are also welcome to include additional files or documentation in the folder or README.md if they help people better understand your project and code.

Step 3: Upload your code
- Commit your changes (no requirements on the commit message)
  - git add .
  - git commit -m "upload project code by Team-XX"
- Push the changes to your fork
  - git push origin main
- On GitHub, navigate to your fork and open a pull request via: Pull requests → New pull request
