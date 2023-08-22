Certainly! Below is a README file that explains the purpose of the program, how to install it, set up the environment, and run it.

---

# UFO Shape Classification Program

## Purpose
This program is designed to classify UFO sighting descriptions into refined shape categories. By taking a CSV file containing descriptions of UFO sightings, it categorizes them into predefined shape classes like "Saucer or Disk," "Triangle," "Cylinder/Cigar," etc. The classification leverages OpenAI's GPT-4 model with detailed classification instructions.
More of a proof of concept for using GPT-4 to classify UFO sightings and has surprisingly good results. Also includes a CLI program to quickly validate the results manually. Data used is included in /data folder.

## Installation and Setup

### Requirements
- Python 3.11
- Required libraries: pydantic, openai, pandas, tiktoken, python-dotenv, logging

### Installation Steps

1. **Install pyenv:**
   If you don't have `pyenv` installed, follow the instructions for your platform [here](https://github.com/pyenv/pyenv#installation).

2. **Install Python 3.11 Using pyenv:**
   Once `pyenv` is installed, you can install Python 3.11 with the following command:

   ```bash
   pyenv install 3.11.0
   ```

3. **Create a Virtual Environment (Optional but Recommended):** 
   Using a virtual environment helps in isolating the dependencies for this project from your global Python environment.

   ```bash
   pyenv virtualenv 3.11.0 my_project_env
   pyenv local my_project_env
   ```

4. **Clone the Repository (if applicable):**
   If this code is hosted in a repository, clone it to your local machine.

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

5. **Install the Required Libraries:**
   The required libraries can be installed using the following command.

   ```bash
   pip install pydantic openai pandas tiktoken python-dotenv
   ```

6. **Set Up Environment Variables:**
   You'll need to set up environment variables for your OpenAI API key and organization. Create a `.env` file in the project directory with the following content:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_ORGANIZATION=your_openai_organization
   ```

   Make sure to replace the placeholders with your actual values.

7. **Data Preparation:**
   Place your cleaned and exported sightings CSV file in the `data/` directory.

8. **Run the Program:**
   You can run the program by executing the main script.

   ```bash
   python main.py
   ```

## How It Works

1. **Preprocessing:** The program reads the UFO sighting descriptions, preprocesses the data, and prepares it for classification.
2. **Tokenizing & Batching:** Descriptions are tokenized and batched to ensure they fit within the token limits for the OpenAI model.
3. **API Calls & Classification:** OpenAI's GPT-4 model is called to classify each description into one of the predefined shape categories.
4. **Retry Mechanism:** If a batch results in empty labels, the program can retry the request a specified number of times.
5. **Logging:** The program includes detailed logging to both a file and the console, making it easier to track the flow and diagnose any issues.
6. **Output:** The classified labels along with the descriptions are saved in a JSON file.


---

## UFO Sightings Label Verification CLI

### Overview

The UFO Sightings Label Verification program is a command-line interface tool designed to allow users to interactively verify and modify the shape categories assigned to UFO sighting descriptions. The program presents the descriptions and existing labels, and the user has the ability to view, modify, and confirm the labels.

### Running the CLI

#### Requirements
- Python 3.11
- Curses library (should be included with Python)

#### Usage

1. **Navigate to the Project Directory:**
   Open a terminal and navigate to the directory containing the CLI script.

2. **Run the CLI Program:**
   Execute the following command, replacing `<file_path>` with the path to the JSON file containing the UFO sighting descriptions and labels.

   ```bash
   python cli_program.py <file_path>
   ```

### Interface

The CLI presents the following information for each UFO sighting:

- **Description:** The description of the UFO sighting.
- **Existing Labels:** The current shape categories assigned to the description.
- **All Categories:** A list of all possible shape categories.

### Interaction

- **Scrolling:** Use the UP and DOWN arrow keys to scroll through the description if it is too long to fit on the screen.
- **Modifying Labels:** At the prompt "Choose categories (e.g., '1 2 3'):", enter the numbers corresponding to the categories you want to add or remove. If a category is already assigned, it will be removed; if not, it will be added.
- **Confirming Changes:** Press the Enter key to confirm the changes and move to the next description.

### Saving

The program saves changes continuously to the JSON file specified in the command line argument.

