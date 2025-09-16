from zenrows import ZenRowsClient
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
import requests, os, re, time
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from langchain_core.documents import Document

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings


load_dotenv()
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
AZURE_ENDPOINT = os.getenv(
    "AZURE_ENDPOINT", "https://pg-mcbw91ko-eastus2.cognitiveservices.azure.com/"
)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# print(AZURE_ENDPOINT, AZURE_OPENAI_API_KEY)
openai_embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small",  # e.g., "text-embedding-ada-002"
    openai_api_key="4eqKbBqorxl3A8LYPxqqUzcTc2LYuDrux5hl6wp9Af0187blgTvzJQQJ99BFACYeBjFXJ3w3AAAAACOG4nXD",
    azure_endpoint="https://narola-ai.cognitiveservices.azure.com/",
    openai_api_version="2024-12-01-preview",  # or the latest supported version
)
pinecone_index_name = "testing"
namespace = "rtc_bot"

pinecone = PineconeClient(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(pinecone_index_name)

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    azure_deployment="gpt-4o",
    api_version="2024-05-01-preview",
    temperature=0.2,
)

Base_url = "https://nowfoodsthailand.com/EN/"
print("hi jency")


# Function to clean extracted text
def clean_text(text):
    if not text:
        return "No text provided to clean"
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove special characters (optional, customize as needed)
    text = re.sub(r"[^\w\s.,!?]", "", text)
    return text if text else "Empty text after cleaning"


# Initialize ZenRows client
client = ZenRowsClient("c80710675940b58f1b6128804cc2f26502aa20c0")
url = "https://nowfoodsthailand.com/EN"
cleaned_text = {}


def zenrows_scraping(url):
    # Configure parameters for HTML content
    params = {
        "js_render": "true",  # Render JavaScript if the page is dynamic
        "premium_proxy": "true",  # Use premium proxies to avoid blocks
        "proxy_country": "us",  # Optional: Set proxy country
        "wait": 10000,  # Wait 10 seconds for JavaScript to load
    }

    # Make the request
    try:
        print("Sending request to:", url)
        response = client.get(url, params=params)
        print("Response status code:", response.status_code)

        # Check if the request was successful
        if response.status_code == 200:
            print("Response received, parsing HTML...")
            # Parse HTML content with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract all text from the page for debugging
            all_text = soup.get_text(strip=True)
            print("Raw extracted text (first 500 characters):", all_text[:500])

            # Extract meaningful text (e.g., paragraphs, headings)
            text_elements = soup.find_all(
                ["p", "h1", "h2", "h3", "div"]
            )  # Customize tags as needed
            if not text_elements:
                print("No elements found with specified tags (p, h1, h2, h3, div)")
            else:
                print(f"Found {len(text_elements)} elements with specified tags")

            raw_text = " ".join(
                element.get_text(strip=True) for element in text_elements
            )
            print(
                "Raw combined text (first 500 characters):",
                raw_text[:500] if raw_text else "No raw text extracted",
            )

            # Clean the text
            cleaned_text = clean_text(raw_text)
            print("Cleaned text:", cleaned_text)
        else:
            print(f"Failed to fetch webpage: Status code {response.status_code}")
            print("Response content (first 500 characters):", response.text[:500])
        return cleaned_text
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""


all_links = [
    "https://www.rtctycoon.com",
    "https://www.rtctycoon.com/Home/64ba246ea7391f2170bd776a/langEN",
    "https://www.rtctycoon.com/About_Und_us/6679201c254e5a00134466d7/langEN",
    "https://www.rtctycoon.com/Trainer/66792022c098220013d97f11/langEN",
    "https://www.rtctycoon.com/Courses/66792028c098220013d97f20/langEN",
    "https://www.rtctycoon.com/Investment_Und_Results/6679202dc098220013d97f2e/langEN",
    "https://www.rtctycoon.com/article/667e740fa80ff4001339d29d/What_Und_is_Und_a_Und_specific_Und_business_Und_tax:!66791ffdb60e9d00133e92b2_LP/667e7367a80ff4001339d28a/66862ecc51db950013c31bbb/langEN",
    "https://www.rtctycoon.com/Contact_Und_us/66792044c098220013d97f47",
]

data = {}
for url in all_links:
    result = zenrows_scraping(url)
    if result:
        data[url] = result

# print("data", data)
output_file_path = "scraped_rtc.txt"

with open(output_file_path, "w", encoding="utf-8") as file:
    for url, content in data.items():
        file.write(f"URL: {url}\n")
        file.write(f"Content:\n{content}\n")
        file.write("=" * 80 + "\n\n")

print(f"\nScraped content saved to {output_file_path}")
documents = [
    Document(page_content=content, metadata={"source": url})
    for url, content in data.items()
]


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

vectorstore = PineconeVectorStore.from_documents(
    texts,
    embedding=openai_embeddings,
    index_name=pinecone_index_name,
    namespace=namespace,
)
