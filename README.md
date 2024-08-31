# TextRank Summarization and Keyword Extraction API

This project is a FastAPI-based web application that implements the TextRank algorithm for text summarization and keyword extraction. The results are displayed on a web interface, with the option to visualize the keyword extracted as a graph.

## Features

- **Text Summarization**: Generates a summary of the input text.
- **Keyword Extraction**: Extracts keywords from the input text and visualizes the relationships in a graph.
- **Graph Visualization**: Displays a graph of the keyword extraction process.
- **Dynamic Web Interface**: Users can input text, select an algorithm, and view results on the same page.

## API Endpoints

- `GET /`: Home page with a form to execute TextRank algorithm.
- `POST /`: Processes the form submission, running the TextRank algorithm based on user input.
- `GET /graph`: Returns the keyword extraction graph as an image.

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- FastAPI
- Uvicorn
- NetworkX
- Matplotlib

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/textrank-api.git
    cd textrank-api
    ```

2. **Create a virtual environment and activate it:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the FastAPI application:**

    ```bash
    uvicorn main:app --reload
    ```

5. **Access the web interface:**

    Open your browser and go to `http://127.0.0.1:8000`.

## Configuration

The application settings are managed in `app/settings/core.py`. You can configure project details like `PROJECT_NAME`, `DESCRIPTION`, and CORS settings there.

## Usage

- **Text Summarization:** Enter text into the input box, select "Summarization," and submit to get a summary.
- **Keyword Extraction:** Enter text, select "Keywords," and submit to extract keywords and view a keyword graph.
