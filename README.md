# Stock Price Prediction with LSTM

This project involves building, training, and deploying a Long Short-Term Memory (LSTM) model to predict stock prices. The project leverages various tools and technologies including TensorFlow, MLFlow, Docker, and Google Cloud.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Setup Instructions](#setup-instructions)
- [Running the Project](#running-the-project)
- [MLFlow Tracking](#mlflow-tracking)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to predict future stock prices using historical stock price data. The model used for this task is a Long Short-Term Memory (LSTM) neural network. The project includes data preprocessing, model training, evaluation, and deployment using Docker and Google Cloud services.

## Directory Structure

The project is organized as follows:

project/
│
├── data/
│ ├── init.py
│ ├── preprocessing.py
│ └── dataset.csv
│
├── models/
│ ├── init.py
│ ├── model.py
│ ├── evaluation.py
│ └── tuner.py
│
├── scripts/
│ ├── init.py
│ ├── train.py
│ └── evaluate.py
│
├── Dockerfile
├── requirements.txt
└── README.md


- **data/**: Contains the data preprocessing scripts and the dataset.
- **models/**: Contains the model architecture, tuning, and evaluation scripts.
- **scripts/**: Contains the training and evaluation scripts.
- **Dockerfile**: Used to build the Docker image for deployment.
- **requirements.txt**: Lists the Python dependencies for the project.
- **README.md**: Documentation for the project.

## Setup Instructions

To set up the project locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   
2. **Create a virtual environment**:
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install dependencies**:
    pip install -r requirements.txt

4. **Set up environment variables (if required)**:
    Create a .env file in the root directory and add any necessary environment variables.

5. **Run the training script**:
     python scripts/train.py

## Running the Project

**Running Locally**

You can run the scripts locally after setting up your environment. For example, to train the model:
python scripts/train.py

**Running with Docker**
Build and run the Docker container:

1. Build the Docker image:
docker build -t finpro .

2. Run the Docker container:
docker run -p 8080:8080 stock-prediction-app

## MLFlow Tracking
This project uses MLFlow to track experiments, including model parameters, metrics, and artifacts. You can start an MLFlow tracking server locally or on a cloud VM.

**To start the MLFlow server locally:**
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
Logs and models from the training process will be stored and accessible via the MLFlow UI at http://localhost:5000.

## Deployment
This project can be deployed using Google Cloud Run or Google Cloud VM.

Deploy to Google Cloud Run
Build the Docker image:

docker build -t gcr.io/[PROJECT-ID]/stock-prediction-app .
Push the Docker image to Google Container Registry:

docker push gcr.io/[PROJECT-ID]/stock-prediction-app
Deploy the image to Google Cloud Run:

gcloud run deploy --image gcr.io/[PROJECT-ID]/stock-prediction-app --platform managed
Deploy to Google Cloud VM
Create and configure a VM on Google Cloud.
SSH into the VM and clone the repository.
Run the training or inference scripts directly on the VM.

## Contributing
If you'd like to contribute to this project, please fork the repository and use a feature branch. Pull requests are warmly welcome.

1. Fork the repo and create your branch:
git checkout -b feature-branch

2. Commit your changes:
git commit -m "Add some feature"

3. Push to the branch:
git push origin feature-branch

4. Submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

### **Penjelasan Setiap Bagian:**

- **Project Overview:** Menjelaskan tujuan proyek, termasuk penggunaan LSTM untuk prediksi harga saham.
- **Directory Structure:** Memberikan gambaran tentang struktur direktori proyek , membantu pengguna memahami di mana mereka dapat menemukan berbagai file dan fungsionalitas.
- **Setup Instructions:** Menyediakan panduan langkah demi langkah untuk mengatur lingkungan proyek dan menjalankan skrip.
- **Running the Project:** Menjelaskan bagaimana menjalankan proyek secara lokal dan menggunakan Docker.
- **MLFlow Tracking:** Menjelaskan cara menggunakan MLFlow untuk melacak eksperimen.
- **Deployment:** Instruksi untuk deploy aplikasi di Google Cloud Run dan Google Cloud VM.
- **Contributing:** Panduan tentang bagaimana pengguna lain dapat berkontribusi pada proyek.
- **License:** Menjelaskan lisensi yang digunakan dalam proyek.
