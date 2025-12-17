# UVA Machine Learning 25F Projects – Submission Guide

This repository contains instructions for submitting your project code to the course repo. Follow the steps below to fork, prepare, and submit your work.

## 1) Set up your local branch
- Fork the course repository: `https://github.com/Qdata4Capstone/uva-machine-learning-25f-projects`
- Clone your fork locally: `git clone https://github.com/<your-username>/uva-machine-learning-25f-projects.git`
- Add the upstream remote: `git remote add upstream https://github.com/Qdata4Capstone/uva-machine-learning-25f-projects.git`

## 2) Prepare your code
In the repo root, create a folder named for your team ID (e.g., `team-1`, `team-11`, `team-111`). Inside that folder include:
- `src/` – all source code.
- `data/` – data required to reproduce results. If data cannot be uploaded, add a markdown file describing how to collect it.
- `requirements.txt` – list of required packages.
- `README.md` – include:
  - Project Title  
  - Team ID and Members  
  - Overview (brief intro to the project)  
  - Usage (how to run the code to get core results)  
  - (Optional) Setup instructions for non-trivial environments  
  - (Optional) Video link with a short description  
- You may add any extra files/docs that help others understand or reproduce your work.

## 3) Upload your code
- Commit your changes: `git add .` then `git commit -m "upload project code by Team-XX"`
- Push to your fork: `git push origin main`
- Open a pull request from your fork to the course repo: GitHub → Pull requests → New pull request.
