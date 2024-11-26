# 004-Reinforcement-Learning

## Description
This project implements a Reinforcement Learning agent designed to trade in a hybrid environment consisting of both real and simulated datasets. Using a Wasserstein Generative Adversarial Network (WGAN), the project generates simulated stock price scenarios to train the agent and avoid overfitting. The agent is evaluated on its ability to optimize rewards by making sequential decisions, balancing risks, and maximizing performance metrics like the *Max Drawdown*.

## Technologies Used
- Python 3.11
- Libraries: NumPy, Pandas, TensorFlow, TA-Lib, Gym, Yahoo Finance

## Features
- *Reinforcement Learning Agent*: Trains on real and simulated datasets.
- *Custom Hybrid Trading Environment*: Built with Gym to simulate realistic trading scenarios.
- *WGAN Simulations*: Generates 1000 synthetic datasets to augment training data.
## Prerequisites
1. Python 3.12 installed on your machine.

## Installation Instructions

## Steps for Windows:
1. Clone the repository:
   bash
   git clone https://github.com/Kike14/004-Reinforcement-Learning

2. Create a virtual environment:
   bash
   python -m venv venv

3. Activate the virtual environment:
   bash
   .\venv\Scripts\activate
4. Upgrade pip:
   bash
   pip install --upgrade pip

5. Install dependencies:
   bash
   pip install -r requirements.txt

6. Run the main script:
   bash
   python main.py

## Steps for Mac:
1. Clone the repository:
   bash
   git clone https://github.com/Kike14/004-Reinforcement-Learning

2. Create a virtual environment:
   bash
   python3 -m venv venv

3. Activate the virtual environment:
   bash
   source venv/bin/activate

4. Upgrade pip:
   bash
   pip install --upgrade pip

5. Install dependencies:
   bash
   pip install -r requirements.txt

6. Run the main script:
   bash
   python main.py 

## Project Structure:
004-Reinforcement-Learning/
│
├── .idea/
├── .venv/
├── .gitignore
├── agent.py
├── data_utils.py
├── env.py
├── generador.keras
├── LICENSE
├── main.py
├── README.md
├── main.py
├──requirements.txt
└── scenario_generator.py

## Contributing
Contributions are welcome. Please submit a pull request following the style guidelines and best practices.

## License
Copy this entire block and paste it into the README.md file in your PyCharm project. This version includes all necessary steps for both Windows and Mac, along with the full project description, structure, and license details.