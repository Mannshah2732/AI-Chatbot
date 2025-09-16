import os, requests, re
from dotenv import load_dotenv
from langchain_core.documents import Document
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import time
from requests_html import HTMLSession
from zenrows import ZenRowsClient


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
  
)

llm = ChatOpenAI(model="gpt-4.1")

pinecone_index_name = "robeck-dental"
namespace = "robeck_dental"

pinecone = PineconeClient(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(pinecone_index_name)


base_url = "https://www.robeckdental.com/"

session = HTMLSession()

client = ZenRowsClient("56e64c44f855866fe895948b65886641be9cfe4f")

def get_internal_links2(base_url):
    visited = set()
    to_visit = [base_url]
    all_links = set()

    while to_visit:
        url = to_visit.pop()
        if url in visited:
            continue
        visited.add(url)
        print("Crawling:", url)

        try:
            response = session.get(url)
            response.html.render(timeout=20)  # render JS
        except Exception as e:
            print(f"Failed: {e}")
            continue

        for link in response.html.absolute_links:
            if base_url in link and urlparse(link).path != "/":
                all_links.add(link)

    return list(all_links)

def get_internal_links(base_url):
    visited = set()
    to_visit = [base_url]
    all_links = set()

    while to_visit:
        url = to_visit.pop()
        if url in visited:
            continue
        visited.add(url)
        print("Crawling:", url)

        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
        except Exception as e:
            print("Failed:", e)
            continue
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)

            # Filter only internal / English links
            if base_url in full_url and "/en/" in full_url:
                cleaned_url = full_url.split("#")[0]  # remove fragment
                if cleaned_url not in visited:
                    to_visit.append(cleaned_url)
                    all_links.add(cleaned_url)
    return list(all_links)

all_links = get_internal_links(base_url)
print("✅ Found Links:", all_links)

internal_links = get_internal_links2(base_url)
print("✅ Found Links 2:", internal_links)

# Function to clean extracted text
def clean_text(text):
    if not text:
        return "No text provided to clean"
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove special characters (optional, customize as needed)
    text = re.sub(r"[^\w\s.,!?]", "", text)
    return text if text else "Empty text after cleaning"

cleaned_text = {}
"""
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
    "https://www.robeckdental.com/",
    "https://www.robeckdental.com/cleaningprevention/",
    "https://www.robeckdental.com/familyservices/",
    "https://www.robeckdental.com/extractionsandpreservation/",
    "https://www.robeckdental.com/cosmeticdentistry/",
    "https://www.robeckdental.com/rootcanals/",
    "https://www.robeckdental.com/cerec/",
    "https://www.robeckdental.com/implants/",
    "https://www.robeckdental.com/invisalign/",
    "https://www.robeckdental.com/oralappliances/",
    "https://www.robeckdental.com/sedationdentistry/",
    "https://www.robeckdental.com/technology/",
    "https://www.robeckdental.com/financial-information/",
    "https://www.robeckdental.com/our-products/",
    "https://www.robeckdental.com/testimonials/",
    "https://www.robeckdental.com/meet-the-team/",
    "https://www.robeckdental.com/post-opcare/",
    "https://www.robeckdental.com/ourhistory/",
    "https://www.robeckdental.com/blog/",
    "https://www.robeckdental.com/contact/",
    "https://www.robeckdental.com/terms-and-conditions/",
    "https://www.robeckdental.com/privacy-policy/",
    "https://www.robeckdental.com/newpatients/",
    "https://www.robeckdental.com/services/"
]

data = {}
for url in all_links:
    result = zenrows_scraping(url)
    if result:
        data[url] = result

# print("data", data)
output_file_path = "scraped_robeck.txt"

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
"""
input_file_path = "scraped_robeck.txt"
with open(input_file_path, "r", encoding="utf-8") as file:
    raw_text = file.read()

# Create a single Document
document = Document(page_content=raw_text)

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents([document])

# Upload to Pinecone
vectorstore = PineconeVectorStore.from_documents(
    texts,
    embedding=openai_embeddings,
    index_name=pinecone_index_name,
    namespace=namespace,
)

print(f"✅ Uploaded {len(texts)} chunks from TXT file to Pinecone namespace '{namespace}'")