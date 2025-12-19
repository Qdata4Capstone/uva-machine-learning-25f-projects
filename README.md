# Students' Final Projects  for 2025 Fall UVa CS -ML-Undergraduate

---

## Related: Students' projects from this course's past offering in 2020 and 2019
- [machine-learning-uva2019-students-deep2reproduce/](https://github.com/Qdata4Capstone/machine-learning-uva19-students-deep2reproduce/)
- [machine-learning-uva2020-students-ml for good use](https://github.com/Qdata4Capstone/machine-learning20f-learning4good-projects)

## The course website : [https://qiyanjun.github.io/2025Fall-UVA-CS-MachineLearningDeep/](https://qiyanjun.github.io/2025Fall-UVA-CS-MachineLearningDeep/)

---

### Index of the students' team projects:  
| Index | Keywords | Video |
|-------|----------|-------|
| t01 | ML_Anomaly | [link](https://youtu.be/lEvAmGlhEuE) |
| t02 | ML_YoloTrash | [link1](https://youtu.be/L-Bif0_NOkU),[link2](https://youtu.be/L-Bif0_NOkU) |
| t03 | ML_gradCam |  |
| t04 | ML_soccerplayer | [link](https://youtu.be/kHUYCva1sxY) |
| t05 | Human-In-the-LoopRL-ImageSynth | [link](https://drive.google.com/file/d/1ztj9GcXTr1e4RqEcebv-Fzs70U8VtJij/view?usp=drive_link) |
| t06 | DiabetesForecasting | [link](https://youtu.be/4sVgWTkxrcI) |
| t07 | Local-RAG-Vector-Search-System | [link](https://www.youtube.com/watch?v=N8b3yETTTbA) |
| t08 | Autodifferentiation | [link](https://youtu.be/1IJK0WWCo3E) |
| t09 | ChessElo | [link](https://youtu.be/SgSRFI_cDYs) |
| t10 | MarketMinds-Headlines-to-Returns | [link](https://youtu.be/iN-jLGnTcNQ) |
| t11 | FairCreditPredictionML |  |
| t12 | ML_major-news |  |
| t13 | CSMedicalImage |  |
| t14 | InventoryMonitor | [link](https://www.youtube.com/watch?v=TyAtKc9BPjs) |
| t15 | ML-skinI | [link](https://youtu.be/dWKxdcSMWO4) |
| t16 | CreditCardFraud | [link](youtube.com/watch?v=O41fVjY96qk&feature=youtu.be) |
| t17 | CSAIassistant | [link](https://youtu.be/IAYtRCeQ7_s) |
| t18 | MicroProgram |  |
| t19 | canvasGPT | [link](https://www.youtube.com/watch?v=IBh6QZ9l06A) |
| t20 | FraudulentAccount | [link](https://youtu.be/lVr0j_KLIIk) |
| t21 | ML_Image_Colorization_Presentation | [link](https://myuva-my.sharepoint.com/:p:/g/personal/utw5es_virginia_edu/IQBKualfcb7LSYgeoj0j1Ww0ARdaqcZu8mnCsosx3DE0vdI?e=xObXdm) |
| t22 | SA-musicians | [link1](https://youtu.be/75aeonsjKK0),[link2](https://youtu.be/ZwsHOmrfXFE),[link3](https://youtu.be/l7HYH53H9nw),[link4](https://youtu.be/GlcwNCH_gfM),[link5](https://youtu.be/BruqunChwj8) |
| t23 | ML_EngineFailure | [link](https://www.youtube.com/watch?v=jBvGLsg9_vc) |
| t24 | StockPrice | [link](https://drive.google.com/file/d/1_knZ4cJ5R_VtIu8JD36ALOirAj8IVv1H/view?usp=sharing) |
| t25 | DrunkDriver |  |
| t26 | MLBrain_tumor |  |
| t27 | ETF-risk | [link](https://youtu.be/KxbXr0T0Rvw) |
| t28 | DJ_Mixing_Recommendation_Final | [link](https://www.youtube.com/watch?v=X2zmxph66xg&feature=youtu.be) |
| t29 | When_Does_ML_Fail_Presentation |  |
| t30 | Mushroom | [link](https://youtu.be/5bpg49MUFBY) |
| t31 | ML-outfit |  |


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
