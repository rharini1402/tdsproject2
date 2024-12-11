import os
import sys
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import chardet 
import io


API_PROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/"
CHAT_COMPLETIONS_ENDPOINT = "chat/completions"
EMBEDDINGS_ENDPOINT = "embeddings"
TOKEN_ENV_VAR = "AIPROXY_TOKEN"
SUPPORTED_CHAT_MODEL = "gpt-4o-mini"
SUPPORTED_EMBEDDING_MODEL = "text-embedding-3-small"


def load_dataset_with_encoding(dataset_path):
    """Load dataset with encoding detection and fallback handling."""
    try:
        
        with open(dataset_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding'] or 'utf-8'  
            print(f"Detected encoding: {encoding}")
        
        
        decoded_content = raw_data.decode(encoding, errors='replace')
        df = pd.read_csv(io.StringIO(decoded_content))
        print("Dataset loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)


def ensure_token():
    """Ensure the AIPROXY_TOKEN is available."""
    token = os.environ.get("AIPROXY_TOKEN")
    if not token:
        print("Error: AIPROXY_TOKEN environment variable is not set.")
        sys.exit(1)
    return token

def fetch_chat_completion(messages, model=SUPPORTED_CHAT_MODEL):
    """Fetch chat completion from AI Proxy."""
    token = ensure_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
    }
    response = requests.post(
        f"{API_PROXY_URL}{CHAT_COMPLETIONS_ENDPOINT}",
        headers=headers,
        json=payload,
    )
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        sys.exit(1)
    return response.json()

def fetch_embeddings(input_texts, model=SUPPORTED_EMBEDDING_MODEL):
    """Fetch embeddings from AI Proxy."""
    token = ensure_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": input_texts,
    }
    response = requests.post(
        f"{API_PROXY_URL}{EMBEDDINGS_ENDPOINT}",
        headers=headers,
        json=payload,
    )
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        sys.exit(1)
    return response.json()


def narrate_story(data, analysis_summary, visualizations):
    """
    Generate a narrative story for the dataset analysis.
    
    Args:
        data (pd.DataFrame): The dataset.
        analysis_summary (str): Summary of the dataset from AI Proxy.
        visualizations (list): List of generated visualization filenames.
    
    Returns:
        str: The narrated story.
    """
    story = "### Dataset Story\n"
    story += "Here is the summary of the dataset:\n\n"
    story += analysis_summary + "\n\n"
    story += "### Visual Insights\n"
    story += "The following visualizations were generated based on the dataset:\n"
    for viz in visualizations:
        story += f"- {viz}\n"
    return story


def generate_visualizations(data):
    """
    Generate visualizations based on the dataset.

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        list: List of generated visualization filenames.
    """
    visualizations = []
    try:
        
        numeric_data = data.select_dtypes(include='number')

        if numeric_data.empty:
            print("No numeric columns available for visualizations.")
            return visualizations

        for column in numeric_data.columns[:2]:  
            plt.figure(figsize=(8, 5))
            numeric_data[column].hist()
            filename = f"{column}_distribution.png"
            plt.title(f"Distribution of {column}")
            plt.savefig(filename)
            visualizations.append(filename)
            plt.close()

     
        if numeric_data.shape[1] > 1:  
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
            filename = "correlation_heatmap.png"
            plt.title("Correlation Heatmap")
            plt.savefig(filename)
            visualizations.append(filename)
            plt.close()

    except Exception as e:
        print(f"Error generating visualizations: {e}")
        sys.exit(1)

    return visualizations



def save_readme(story, visualizations):
    """Save the story and visualizations to README.md."""
    try:
        with open("README.md", "w") as f:
            f.write("# Analysis Report\n\n")
            f.write("## Generated Story\n")
            f.write(story + "\n\n")
            f.write("## Visualizations\n")
            for viz in visualizations:
                f.write(f"![{viz}]({viz})\n")
        print("README.md created successfully.")
    except Exception as e:
        print(f"Error saving README.md: {e}")


def analyze_dataset(dataset_path):
    """Analyze dataset and provide insights."""
    
    df = load_dataset_with_encoding(dataset_path)
    
    
    summary = df.describe(include='all').to_string()
    messages = [{"role": "user", "content": f"Summarize this dataset for storytelling:\n\n{summary}"}]
    response = fetch_chat_completion(messages)

    
    return df, response.get("choices", [{}])[0].get("message", {}).get("content", "No response generated.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file {dataset_path} not found.")
        sys.exit(1)

    print(f"Analyzing dataset: {dataset_path}")
    data, analysis_summary = analyze_dataset(dataset_path)

   
    print("Generating visualizations...")
    visualizations = generate_visualizations(data)

    
    print("Generating story...")
    story = narrate_story(data, analysis_summary, visualizations)

    
    print("Saving README.md and visualizations...")
    save_readme(story, visualizations)

    print(f"Analysis complete for {dataset_path}. Check the README.md and generated images.")


if __name__ == "__main__":
    main()
