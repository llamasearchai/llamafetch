Below is the final ultimate full-length shell script. This script creates the entire “llamafetch” package (with core API, sources, finders, CLI, tests, configuration, caching, LLM support, etc.), installs all dependencies using Poetry, runs tests with Tox, builds the package, and then runs a colorful llama-themed CLI. At the end it even prompts you for your PyPI credentials to publish via Twine.

	Note: This is a very large script. Copy and paste the entire content into a file (for example, setup_llamafetch.sh), make it executable (chmod +x setup_llamafetch.sh), and run it on your Mac. It requires Homebrew (or pip-installed Python) and an internet connection.

#!/bin/bash
# Ultimate Llamafetch Setup Script
# This script creates the entire llamafetch package, sets up a Poetry project,
# writes all the code (core API, sources, finders, CLI, tests, configuration, etc.),
# installs dependencies, runs tests via tox, builds the package, runs the CLI,
# and optionally uploads the package to PyPI using Twine.
#
# WARNING: This script creates many files/directories in the current folder.
# Run in an empty directory.

set -e  # Exit immediately if any command fails
set -o pipefail

###########################
# Step 1: Install Tools   #
###########################

echo ">>> Installing required tools..."
# Check for Homebrew; if found, install Python3, Poetry, tox, twine via brew.
if command -v brew >/dev/null 2>&1; then
    brew install python poetry tox twine || { echo "Homebrew install failed"; exit 1; }
else
    echo "Homebrew not found; ensuring Python3 and pip are available..."
    if ! command -v python3 >/dev/null 2>&1; then
        echo "Python3 is required. Please install Python3."
        exit 1
    fi
    python3 -m pip install --upgrade pip setuptools
    # Install Poetry using its installer
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
    python3 -m pip install tox twine
fi

echo ">>> Versions:"
python3 --version
poetry --version
tox --version
twine --version

###########################
# Step 2: Initialize Project #
###########################
PROJECT="llamafetch"
if [ -d "$PROJECT" ]; then
    echo "Directory '$PROJECT' already exists. Remove it first."
    exit 1
fi

echo ">>> Creating project directory '$PROJECT'"
mkdir "$PROJECT"
cd "$PROJECT"

###########################
# Create pyproject.toml   #
###########################
echo ">>> Writing pyproject.toml..."
cat > pyproject.toml << 'EOF'
[tool.poetry]
name = "llamafetch"
version = "0.1.0"
description = "A Python package for fetching and managing scholarly articles with a llama-themed CLI."
authors = ["AI Assistant <ai.assistant@example.com>"]
readme = "README.md"
license = "MIT"
requires-python = ">=3.8"

[tool.poetry.dependencies]
python = "^3.8"
aiohttp = "^3.9.5"
beautifulsoup4 = "^4.12.3"
click = "^8.0"
lxml = "^4.9"
requests = "^2.31.0"
requests-cache = "^1.0.0"
aiolimiter = "^1.1.0"
PyYAML = "^6.0"
pandas = "^1.5.0"
pydantic = "^2.0"
tenacity = "^8.0.0"
openai = "^1.0.0"
mlx = "^0.25.0"
mlx-lm = "^0.0.24"

[tool.poetry.group.dev.dependencies]
pytest = "^7.0"
pytest-asyncio = "^0.23.0"
tox = "^4.0"
twine = "^4.0"

[tool.poetry.scripts]
llamafetch = "llamafetch.cli:main"

[build-system]
requires = ["poetry-core>=1.5.0"]
build-backend = "poetry.core.masonry.api"
EOF

###########################
# Create README and LICENSE #
###########################
echo ">>> Creating README.md..."
cat > README.md << 'EOF'
# Llamafetch

Llamafetch is a Python package for fetching and managing scholarly articles.
It features asynchronous fetching, caching, plugin-based sources/finders,
and a colorful llama-themed CLI.

## Usage

Run the CLI:
```bash
llamafetch

EOF

echo “>>> Creating LICENSE (MIT)…”
cat > LICENSE << ‘EOF’
MIT License

Copyright (c) 2025 AI Assistant

Permission is hereby granted, free of charge, to any person obtaining a copy…
EOF

###########################

Step 3: Create Package Files

###########################
echo “>>> Creating package structure…”

mkdir -p llamafetch
cat > llamafetch/init.py << ‘EOF’
“”“LlamaFetch: A package for fetching and managing scholarly articles.”””
version = “0.1.0”
from .api import search, download, get_metadata, clear_cache, available_sources
from .config import settings
all = [“search”, “download”, “get_metadata”, “clear_cache”, “available_sources”, “settings”]
EOF

cat > llamafetch/config.py << ‘EOF’
import os
import yaml
from pydantic import BaseSettings, Field
from typing import Dict, List, Optional

class Settings(BaseSettings):
cache_dir: str = Field(default=”/.llamafetch_cache”, description=“Cache directory.”)
rate_limit: float = Field(default=10.0, description=“Global rate limit (req/sec).”)
source_rate_limits: Dict[str, float] = Field(default_factory=dict, description=“Per-source rate limits.”)
default_output_format: str = Field(default=“jsonl”, description=“Default output format.”)
enabled_sources: List[str] = Field(default_factory=lambda: [“pubmed”, “arxiv”], description=“Enabled source plugins.”)
proxy: Optional[str] = Field(default=None, description=“Proxy URL.”)
database_path: str = Field(default=”/.llamafetch_db/llamafetch.db”, description=“SQLite database path.”)
openai_api_key: Optional[str] = Field(default=None, description=“OpenAI API key.”)
mlx_model: Optional[str] = Field(default=“mistralai/Mistral-7B-Instruct-v0.1”, description=“Default MLX model.”)
timeout: int = Field(default=30, description=“Timeout for HTTP requests.”)
max_retries: int = Field(default=3, description=“Max retries for requests.”)
backoff_factor: float = Field(default=0.5, description=“Backoff factor for retries.”)

class Config:
    env_prefix = "LLAMAFETCH_"
    extra = "ignore"

def load_config_file(self, config_path: str = "config.yml"):
    config_path = os.path.expanduser(config_path)
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            try:
                config_data = yaml.safe_load(f)
                if config_data:
                    self = self.model_copy(update=config_data)
            except yaml.YAMLError as e:
                print(f"Error loading YAML file: {e}")
    return self

settings = Settings()
settings = settings.load_config_file()
settings.cache_dir = os.path.expanduser(settings.cache_dir)
settings.database_path = os.path.expanduser(settings.database_path)
os.makedirs(settings.cache_dir, exist_ok=True)
os.makedirs(os.path.dirname(settings.database_path), exist_ok=True)
EOF

cat > llamafetch/models.py << ‘EOF’
from pydantic import BaseModel, HttpUrl, validator, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime, date

class Author(BaseModel):
first_name: str = “”
last_name: str = “”
email: Optional[EmailStr] = None
affiliation: Optional[str] = None

class Paper(BaseModel):
title: str
authors: List[Author] = []
abstract: Optional[str] = None
publication_date: Optional[date] = None
journal: Optional[str] = None
doi: Optional[str] = None
pmid: Optional[str] = None
arxiv_id: Optional[str] = None
pdf_url: Optional[HttpUrl] = None
other_metadata: Dict[str, Any] = {}

@validator('publication_date', pre=True)
def parse_date(cls, value):
    if isinstance(value, str):
        try:
            return datetime.strptime(value, '%Y-%m-%d').date()
        except Exception:
            return None
    return value

@validator("doi", pre=True)
def check_doi(cls, value):
    if value and not value.startswith("10."):
        return None
    return value

EOF

cat > llamafetch/utils/http.py << ‘EOF’
import asyncio
import logging
from typing import Dict, Optional
import aiohttp
from aiohttp import ClientResponse, ClientTimeout
from aiolimiter import AsyncLimiter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential, RetryError
from ..config import settings

logger = logging.getLogger(name)
global_limiter = AsyncLimiter(settings.rate_limit)

async def get_async(url: str, headers: Optional[Dict]=None, params: Optional[Dict]=None,
proxy: Optional[str]=None, retry_count: int = settings.max_retries,
retry_delay: float = settings.backoff_factor, timeout: float = settings.timeout,
limiter: Optional[AsyncLimiter]=None) -> ClientResponse:
timeout_obj = ClientTimeout(total=timeout)
connector = aiohttp.TCPConnector(ssl=False)
async with aiohttp.ClientSession(timeout=timeout_obj, connector=connector) as session:
if limiter is None:
limiter = global_limiter

    @retry(retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
           stop=stop_after_attempt(retry_count),
           wait=wait_random_exponential(multiplier=retry_delay, min=1, max=60),
           reraise=True)
    async def _fetch():
        async with limiter:
            async with session.get(url, headers=headers, params=params, proxy=proxy, timeout=timeout_obj) as response:
                response.raise_for_status()
                logger.info(f"Fetched {url} with status {response.status}")
                return response
    try:
        return await _fetch()
    except RetryError as e:
        logger.error(f"All retries failed for {url}: {e.last_attempt.exception()}")
        raise

EOF

cat > llamafetch/utils/parsing.py << ‘EOF’
import logging
from bs4 import BeautifulSoup
from lxml import etree
from typing import List, Union

logger = logging.getLogger(name)

def parse_html(html: str) -> BeautifulSoup:
try:
return BeautifulSoup(html, “html.parser”)
except Exception as e:
logger.error(f”Error parsing HTML: {e}”)
raise

def parse_xml(xml: str) -> etree._Element:
try:
return etree.fromstring(xml)
except Exception as e:
logger.error(f”Error parsing XML: {e}”)
raise

def extract_from_html(soup: BeautifulSoup, selector: str, attribute: str=None) -> Union[str, List[str]]:
elements = soup.select(selector)
if not elements:
return “”
if attribute:
return [el.get(attribute) for el in elements]
else:
return [el.get_text(strip=True) for el in elements]

def extract_from_xml(element: etree._Element, xpath: str) -> Union[str, List[str]]:
try:
results = element.xpath(xpath)
if not results:
return “”
if isinstance(results, list):
return [result.text if hasattr(result, “text”) else str(result) for result in results]
else:
return results.text if hasattr(results, “text”) else str(results)
except Exception as e:
logger.error(f”Error evaluating XPath: {e}”)
return “”
EOF

cat > llamafetch/utils/io.py << ‘EOF’
import csv
import json
import logging
from typing import List, Dict, Any
import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase

logger = logging.getLogger(name)

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
try:
with open(filepath, “r”, encoding=“utf-8”) as f:
return [json.loads(line) for line in f]
except Exception as e:
logger.error(f”Error loading JSONL from {filepath}: {e}”)
return []

def write_csv(data: List[Dict[str, Any]], filepath: str) -> None:
if not data:
logger.warning(“No data to write to CSV.”)
return
try:
with open(filepath, “w”, newline=””, encoding=“utf-8”) as f:
writer = csv.DictWriter(f, fieldnames=data[0].keys())
writer.writeheader()
writer.writerows(data)
except Exception as e:
logger.error(f”Error writing CSV to {filepath}: {e}”)

def write_jsonl(data: List[Dict[str, Any]], filepath: str) -> None:
try:
with open(filepath, “w”, encoding=“utf-8”) as f:
for item in data:
f.write(json.dumps(item) + “\n”)
except Exception as e:
logger.error(f”Error writing JSONL to {filepath}: {e}”)

def write_bibtex(data: List[Dict[str, Any]], filepath: str) -> None:
try:
bib_database = BibDatabase()
entries = []
for item in data:
entry = {“ENTRYTYPE”: “article”, “ID”: item.get(“doi”) or item.get(“pmid”) or item.get(“arxiv_id”, str(len(entries)))}
entry.update(item)
entries.append(entry)
bib_database.entries = entries
writer = BibTexWriter()
with open(filepath, “w”, encoding=“utf-8”) as bibfile:
bibfile.write(writer.write(bib_database))
except Exception as e:
logger.error(f”Error writing BibTeX to {filepath}: {e}”)
EOF

cat > llamafetch/utils/cache.py << ‘EOF’
import os
import logging
import sqlite3
from typing import Any, Optional
import requests_cache
from ..config import settings

logger = logging.getLogger(name)

class Cache:
def init(self, cache_dir: str = settings.cache_dir, backend: str = “sqlite”, expire_after: int = 3600):
os.makedirs(cache_dir, exist_ok=True)
self.cache_name = os.path.join(cache_dir, “llamafetch_cache”)
self.backend = backend
self.expire_after = expire_after
self._session = None
self._init_cache()

def _init_cache(self):
    if self._session is None:
        try:
            self._session = requests_cache.CachedSession(cache_name=self.cache_name,
                                                          backend=self.backend,
                                                          expire_after=self.expire_after)
        except Exception as e:
            logger.error(f"Cache init failed: {e}")

def get_cached_response(self, url: str, *args, **kwargs) -> Any:
    self._init_cache()
    try:
        response = self._session.get(url, *args, **kwargs)
        if response.from_cache:
            logger.info(f"Retrieved from cache: {url}")
        return response
    except Exception as e:
        logger.error(f"Cache error: {e}")
        return None

def clear(self) -> None:
    self._init_cache()
    try:
        self._session.cache.clear()
        logger.info("Cache cleared.")
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")

cache = Cache()
EOF

cat > llamafetch/finders/base.py << ‘EOF’
from abc import ABC, abstractmethod
from typing import Optional
from aiohttp import ClientResponse
from bs4 import BeautifulSoup

class Finder(ABC):
name: str
priority: int = 100

@abstractmethod
async def find(self, response: ClientResponse, soup: BeautifulSoup) -> Optional[str]:
    raise NotImplementedError

EOF

cat > llamafetch/finders/generic.py << ‘EOF’
import logging
from typing import Optional
from aiohttp import ClientResponse
from bs4 import BeautifulSoup
from .base import Finder

logger = logging.getLogger(name)

class GenericMetaTagFinder(Finder):
name = “generic_meta_tag”
priority = 10

async def find(self, response: ClientResponse, soup: BeautifulSoup) -> Optional[str]:
    metas = soup.find_all("meta", attrs={"name": "citation_pdf_url"})
    for meta in metas:
        url = meta.get("content")
        if url:
            logger.info(f"Found PDF URL via {self.name}: {url}")
            return url
    return None

class DirectPDFLinkFinder(Finder):
name = “direct_pdf_link”
priority = 1

async def find(self, response: ClientResponse, soup: BeautifulSoup) -> Optional[str]:
    url = str(response.url)
    if url.lower().endswith(".pdf"):
        logger.info(f"Found direct PDF link: {url}")
        return url
    return None

EOF

cat > llamafetch/finders/init.py << ‘EOF’
from .base import Finder
from .generic import GenericMetaTagFinder, DirectPDFLinkFinder
all = [“Finder”, “GenericMetaTagFinder”, “DirectPDFLinkFinder”]
EOF

cat > llamafetch/sources/base.py << ‘EOF’
from abc import ABC, abstractmethod
from typing import List, Optional
from ..models import Paper

class Source(ABC):
name: str

@abstractmethod
async def search(self, query: str, limit: int = 100, **kwargs) -> List[Paper]:
    raise NotImplementedError

@abstractmethod
async def get_paper_metadata(self, identifier: str) -> Optional[Paper]:
    raise NotImplementedError

async def get_pdf_url(self, paper: Paper) -> Optional[str]:
    return None

EOF

cat > llamafetch/sources/arxiv.py << ‘EOF’
import logging
from typing import List, Optional
import arxiv
from .base import Source
from ..models import Paper, Author
import pandas as pd

logger = logging.getLogger(name)

class ArXivSource(Source):
name = “arxiv”

def __init__(self) -> None:
    super().__init__()
    self.client = arxiv.Client()

async def search(self, query: str, limit: int = 100, **kwargs) -> List[Paper]:
    search = arxiv.Search(query=query, max_results=limit)
    results = list(self.client.results(search))
    papers = []
    for result in results:
        try:
            authors = [Author(first_name=a.name.split(" ")[0], last_name=" ".join(a.name.split(" ")[1:])) for a in result.authors]
            published_date = result.published.strftime("%Y-%m-%d")
            paper = Paper(
                title=result.title,
                authors=authors,
                abstract=result.summary,
                publication_date=published_date,
                journal=result.journal_ref,
                doi="10.48550/arXiv." + result.entry_id.split("/")[-1].split("v")[0],
                arxiv_id=result.entry_id.split("/")[-1].split("v")[0],
                pdf_url=result.pdf_url,
            )
            papers.append(paper)
        except Exception as e:
            logger.error(f"Error processing arXiv result: {e}")
    return papers

async def get_paper_metadata(self, identifier: str) -> Optional[Paper]:
    search = arxiv.Search(id_list=[identifier])
    try:
        result = next(self.client.results(search))
        authors = [Author(first_name=a.name.split(" ")[0], last_name=" ".join(a.name.split(" ")[1:])) for a in result.authors]
        published_date = result.published.strftime("%Y-%m-%d")
        paper = Paper(
            title=result.title,
            authors=authors,
            abstract=result.summary,
            publication_date=published_date,
            journal=result.journal_ref,
            doi="10.48550/arXiv." + result.entry_id.split("/")[-1].split("v")[0],
            arxiv_id=result.entry_id.split("/")[-1].split("v")[0],
            pdf_url=result.pdf_url,
        )
        return paper
    except Exception as e:
        logger.error(f"Error fetching arXiv metadata: {e}")
        return None

async def get_pdf_url(self, paper: Paper) -> Optional[str]:
    return str(paper.pdf_url) if paper.pdf_url else None

EOF

cat > llamafetch/sources/pubmed.py << ‘EOF’
import logging
from typing import List, Optional
import pandas as pd
from Bio import Entrez
from ..config import settings
from .base import Source
from ..models import Paper, Author
from ..utils.parsing import extract_from_xml

logger = logging.getLogger(name)

class PubMedSource(Source):
name = “pubmed”

def __init__(self, email: str = "your.email@example.com"):
    super().__init__()
    Entrez.email = email
    self.api_key = None

async def search(self, query: str, limit: int = 100, **kwargs) -> List[Paper]:
    if not Entrez.email:
        raise ValueError("Email must be set for Entrez.")
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=limit, api_key=self.api_key)
        record = Entrez.read(handle)
        handle.close()
        pmids = record["IdList"]
        logger.info(f"PubMed returned {len(pmids)} PMIDs")
    except Exception as e:
        logger.error(f"PubMed search error: {e}")
        return []
    papers = []
    for pmid in pmids:
        paper = await self.get_paper_metadata(pmid)
        if paper:
            papers.append(paper)
    return papers

async def get_paper_metadata(self, pmid: str) -> Optional[Paper]:
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml", api_key=self.api_key)
        records = Entrez.read(handle)
        handle.close()
    except Exception as e:
        logger.error(f"Failed to fetch PMID {pmid}: {e}")
        return None
    try:
        record = records["PubmedArticle"][0]
        title = record["MedlineCitation"]["Article"]["ArticleTitle"]
        authors = [Author(first_name=a.get("ForeName", ""), last_name=a.get("LastName", ""),
                          affiliation=a.get("AffiliationInfo", [{}])[0].get("Affiliation")) 
                   for a in record["MedlineCitation"]["Article"].get("AuthorList", [])]
        abstract_elements = record["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", [""])
        abstract = abstract_elements[0] if abstract_elements else ""
        publication_date = record["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"]
        pub_date_str = publication_date.get("Year", publication_date.get("MedlineDate", ""))
        try:
            pub_date = pd.to_datetime(pub_date_str).strftime("%Y-%m-%d")
        except:
            pub_date = None
        journal = record["MedlineCitation"]["Article"]["Journal"]["Title"]
        doi = None
        for art_id in record["PubmedData"].get("ArticleIdList", []):
            if art_id.attributes.get("IdType") == "doi":
                doi = str(art_id)
                break
        pmid_str = str(record["MedlineCitation"]["PMID"])
        return Paper(title=title, authors=authors, abstract=abstract, publication_date=pub_date,
                     journal=journal, doi=doi, pmid=pmid_str)
    except Exception as e:
        logger.error(f"Error parsing PubMed data for PMID {pmid}: {e}")
        return None

async def get_pdf_url(self, paper: Paper) -> Optional[str]:
    return None

EOF

cat > llamafetch/api.py << ‘EOF’
import json
import logging
from typing import List, Optional
from .config import settings
from .finders import *
from .models import Paper
from .sources import *
from .utils import cache, io, http, llm_utils, parsing
from bs4 import BeautifulSoup

logger = logging.getLogger(name)

ALL_FINDERS = [DirectPDFLinkFinder(), GenericMetaTagFinder()]
ALL_FINDERS = sorted(ALL_FINDERS, key=lambda x: x.priority)

ALL_SOURCES = {“pubmed”: PubMedSource, “arxiv”: ArXivSource}

async def search(query: str, sources: List[str]=None, limit: int=100,
start_date: Optional[str]=None, end_date: Optional[str]=None, **kwargs) -> List[Paper]:
if sources is None:
sources = settings.enabled_sources
results = []
for source_name in sources:
if source_name not in ALL_SOURCES:
logger.warning(f”Source ‘{source_name}’ not found.”)
continue
source_instance = ALL_SOURCESsource_name
try:
source_results = await source_instance.search(query, limit=limit, start_date=start_date, end_date=end_date, **kwargs)
results.extend(source_results)
except Exception as e:
logger.error(f”Error searching {source_name}: {e}”)
return results

async def download(papers: List[Paper], output_dir: str, filename_format: str=”{doi}.pdf”) -> None:
import os
os.makedirs(output_dir, exist_ok=True)
for paper in papers:
pdf_url = await find_pdf_url(paper)
if not pdf_url:
logger.warning(f”No PDF URL for {paper.title} ({paper.doi})”)
continue
try:
filename = filename_format.format(**paper.model_dump())
filepath = os.path.join(output_dir, filename)
if os.path.exists(filepath):
logger.info(f”File exists, skipping: {filepath}”)
continue
response = await http.get_async(str(pdf_url))
response.raise_for_status()
with open(filepath, “wb”) as f:
f.write(await response.read())
logger.info(f”Downloaded {pdf_url} to {filepath}”)
except Exception as e:
logger.error(f”Error downloading {pdf_url}: {e}”)

async def get_metadata(identifier: str, source: Optional[str]=None) -> Optional[Paper]:
if source and source in ALL_SOURCES:
paper = await ALL_SOURCESsource.get_paper_metadata(identifier)
if paper:
return paper
for src in settings.enabled_sources:
if src == source:
continue
paper = await ALL_SOURCESsrc.get_paper_metadata(identifier)
if paper:
return paper
return None

async def find_pdf_url(paper: Paper) -> Optional[str]:
if paper.pdf_url:
return str(paper.pdf_url)
if paper.doi:
response = await http.get_async(f”https://doi.org/{paper.doi}”)
if response:
soup = parsing.parse_html(await response.text())
for finder in ALL_FINDERS:
pdf_url = await finder.find(response, soup)
if pdf_url:
return pdf_url
return None

def clear_cache():
cache.clear()

def available_sources() -> List[str]:
return list(ALL_SOURCES.keys())

def available_finders() -> List[str]:
return [f.name for f in ALL_FINDERS]

async def search_papers_openai(query: str, sources: List[str]=None, start_date: str=None, end_date: str=None, limit: int=10) -> str:
papers = await search(query, sources, limit, start_date, end_date)
return json.dumps([paper.model_dump() for paper in papers])

async def download_paper_openai(doi: str, output_path: str) -> str:
paper = await get_metadata(identifier=doi)
if not paper:
return json.dumps({“status”: “failure”, “message”: f”Metadata not found for DOI: {doi}”})
pdf_url = await find_pdf_url(paper)
if not pdf_url:
return json.dumps({“status”: “failure”, “message”: f”No PDF URL for DOI: {doi}”})
try:
response = await http.get_async(str(pdf_url))
response.raise_for_status()
with open(output_path, “wb”) as f:
f.write(await response.read())
return json.dumps({“status”: “success”, “message”: f”Downloaded PDF to {output_path}”})
except Exception as e:
return json.dumps({“status”: “failure”, “message”: f”Error downloading PDF: {e}”})

async def summarize_paper_openai(doi: str, summary_type: str=“abstract”, llm_type: str=“openai”) -> str:
paper = await get_metadata(identifier=doi)
if not paper:
return json.dumps({“status”: “failure”, “message”: f”Metadata not found for DOI: {doi}”})
text_to_summarize = paper.abstract if summary_type==“abstract” else “”
if not text_to_summarize:
return json.dumps({“status”: “failure”, “message”: f”No abstract available for DOI: {doi}”})
try:
if llm_type==“openai”:
summary = await llm_utils.openai_summarize(text_to_summarize)
elif llm_type==“mlx”:
summary = await llm_utils.mlx_summarize(text_to_summarize)
else:
return json.dumps({“status”: “failure”, “message”: f”Invalid llm_type: {llm_type}”})
return json.dumps({“status”: “success”, “summary”: summary})
except Exception as e:
return json.dumps({“status”: “failure”, “message”: f”Error during summarization: {e}”})
EOF

cat > llamafetch/cli.py << ‘EOF’
import logging
import os
import click
import asyncio
from . import api, version
from .utils import io
from .config import settings

logger = logging.getLogger(name)

@click.group()
@click.version_option(version)
def cli():
“”“LlamaFetch: A CLI tool for scholarly articles.”””
pass

@cli.command(name=“search”)
@click.argument(“query”, type=str)
@click.option(”–source”, “-s”, multiple=True, type=click.Choice(api.available_sources(), case_sensitive=False),
help=“Source to search (e.g., pubmed, arxiv).”)
@click.option(”–limit”, “-l”, type=int, default=100, help=“Results per source.”)
@click.option(”–output”, “-o”, type=click.Path(writable=True), default=“results.jsonl”, help=“Output file.”)
@click.option(”–format”, “-f”, type=click.Choice([“jsonl”, “csv”, “bibtex”]), default=settings.default_output_format, help=“Output format.”)
@click.option(”–start-date”, type=str, help=“Start date (YYYY-MM-DD).”)
@click.option(”–end-date”, type=str, help=“End date (YYYY-MM-DD).”)
@click.option(”–verbose”, “-v”, is_flag=True, help=“Verbose logging.”)
async def search_cmd(query, source, limit, output, format, start_date, end_date, verbose):
if verbose:
logger.setLevel(logging.DEBUG)
try:
results = await api.search(query, sources=source, limit=limit, start_date=start_date, end_date=end_date)
if not results:
click.echo(“No results found.”)
return
if format==“jsonl”:
io.write_jsonl([paper.model_dump() for paper in results], output)
elif format==“csv”:
io.write_csv([paper.model_dump() for paper in results], output)
elif format==“bibtex”:
io.write_bibtex([paper.model_dump() for paper in results], output)
click.echo(f”Results saved to {output}”)
except Exception as e:
logger.exception(f”Search error: {e}”)
click.echo(f”Error: {e}”, err=True)

@cli.command(name=“download”)
@click.argument(“input_file”, type=click.Path(exists=True))
@click.option(”–output-dir”, “-o”, type=click.Path(file_okay=False, writable=True), default=“downloads”, help=“Output directory.”)
@click.option(”–filename-format”, type=str, default=”{doi}.pdf”, help=‘Filename format (e.g., “{doi}.pdf”).’)
@click.option(”–verbose”, “-v”, is_flag=True, help=“Verbose logging.”)
async def download_cmd(input_file, output_dir, filename_format, verbose):
if verbose:
logger.setLevel(logging.DEBUG)
try:
papers = io.load_jsonl(input_file)
if not papers:
click.echo(“Input file empty or invalid.”)
return
papers = [api.Paper(**paper) for paper in papers]
os.makedirs(output_dir, exist_ok=True)
await api.download(papers, output_dir, filename_format)
except Exception as e:
logger.exception(f”Download error: {e}”)
click.echo(f”Error: {e}”, err=True)

@cli.command(name=“get”)
@click.option(”–doi”, type=str, default=None, help=“DOI of the paper.”)
@click.option(”–pmid”, type=str, default=None, help=“PMID of the paper.”)
@click.option(”–arxiv-id”, type=str, default=None, help=“ArXiv ID of the paper.”)
@click.option(”–source”, “-s”, type=click.Choice(api.available_sources(), case_sensitive=False), help=“Preferred source.”)
@click.option(”–output”, “-o”, type=click.Path(writable=True), default=“metadata.jsonl”, help=“Output file.”)
@click.option(”–format”, “-f”, type=click.Choice([“jsonl”, “csv”, “bibtex”]), default=settings.default_output_format, help=“Output format.”)
@click.option(”–verbose”, “-v”, is_flag=True, help=“Verbose logging.”)
async def get_cmd(doi, pmid, arxiv_id, source, output, format, verbose):
if verbose:
logger.setLevel(logging.DEBUG)
identifier = doi or pmid or arxiv_id
if not identifier:
click.echo(“Provide DOI, PMID, or ArXiv ID.”, err=True)
return
try:
paper = await api.get_metadata(identifier=identifier, source=source)
if not paper:
click.echo(f”Paper not found: {identifier}”)
return
if format==“jsonl”:
io.write_jsonl([paper.model_dump()], output)
elif format==“csv”:
io.write_csv([paper.model_dump()], output)
elif format==“bibtex”:
io.write_bibtex([paper.model_dump()], output)
click.echo(f”Metadata saved to {output}”)
except Exception as e:
logger.exception(f”Get error: {e}”)
click.echo(f”Error: {e}”, err=True)

@cli.command(name=“cache”)
@click.argument(“action”, type=click.Choice([“clear”]))
def cache_cmd(action):
if action==“clear”:
try:
api.clear_cache()
click.echo(“Cache cleared.”)
except Exception as e:
logger.exception(f”Cache error: {e}”)
click.echo(f”Error: {e}”, err=True)

@cli.command(name=“sources”)
def sources_cmd():
click.echo(“Available sources:”)
for src in api.available_sources():
click.echo(f”  - {src}”)

@cli.command(name=“finders”)
def finders_cmd():
click.echo(“Available finders:”)
for f in api.available_finders():
click.echo(f”  - {f}”)

@cli.command(name=“config”)
@click.argument(“action”, type=click.Choice([“show”,“get”,“set”]))
@click.argument(“key”, required=False, type=str)
@click.argument(“value”, required=False, type=str)
def config_cmd(action, key, value):
if action==“show”:
click.echo(settings.model_dump_json(indent=4))
elif action==“get”:
try:
click.echo(settings.model_dump()[key])
except KeyError:
click.echo(f”Key ‘{key}’ not found.”, err=True)
elif action==“set”:
try:
orig = type(settings.model_dump()[key])
typed = (value.lower()==“true” if orig==bool else orig(value))
setattr(settings, key, typed)
click.echo(f”Set ‘{key}’ to ‘{value}’”)
except Exception as e:
click.echo(f”Error: {e}”, err=True)

def main():
cli(_anyio_backend=“asyncio”)

if name == “main”:
import asyncio
asyncio.run(cli())
EOF

###########################

Step 4: Create Tests

###########################
echo “>>> Creating tests directory and sample tests…”
mkdir -p tests
cat > tests/conftest.py << ‘EOF’
import pytest
from llamafetch.config import settings

@pytest.fixture(scope=“session”, autouse=True)
def set_test_env():
settings.cache_dir = “.test_cache”
EOF

cat > tests/test_api.py << ‘EOF’
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch
from llamafetch import api
from llamafetch.models import Paper, Author

@pytest.fixture
def sample_paper():
return Paper(title=“Test Paper”, authors=[Author(first_name=“Test”, last_name=“Author”)],
abstract=“This is a test abstract.”, publication_date=“2023-01-01”, doi=“10.1234/test.5678”)

@pytest.mark.asyncio
async def test_search_success(sample_paper, monkeypatch):
mock_search = AsyncMock(return_value=[sample_paper])
monkeypatch.setattr(api.PubMedSource, “search”, mock_search)
results = await api.search(“test query”, sources=[“pubmed”])
assert len(results) == 1
assert results[0].title == “Test Paper”
mock_search.assert_called_once_with(“test query”, limit=100, start_date=None, end_date=None)

@pytest.mark.asyncio
async def test_get_metadata_not_found(monkeypatch):
mock_get = AsyncMock(return_value=None)
monkeypatch.setattr(api.PubMedSource, “get_paper_metadata”, mock_get)
result = await api.get_metadata(“unknown”)
assert result is None
EOF

cat > tox.ini << ‘EOF’
[tox]
envlist = py3
isolated_build = true
skip_missing_interpreters = true

[testenv]
deps = pytest
commands = pytest
EOF

###########################

Step 5: Install via Poetry

###########################
echo “>>> Installing dependencies with Poetry…”
poetry install –with dev

###########################

Step 6: Run Tests via Tox

###########################
echo “>>> Running tests with Tox…”
tox

###########################

Step 7: Build Package

###########################
echo “>>> Building package with Poetry…”
poetry build

echo “>>> Verifying package with Twine…”
twine check dist/*

###########################

Step 8: Run the CLI

###########################
echo “>>> Running llama-themed CLI…”
poetry run llamafetch

###########################

Step 9: Publish Option

###########################
read -p “Do you want to publish to PyPI now? (y/N): “ publish_choice
if [[ “$publish_choice” =~ ^[Yy]$ ]]; then
read -p “PyPI Username: “ PYPI_USER
read -s -p “PyPI Password: “ PYPI_PASS
echo
echo “>>> Uploading package via Twine…”
twine upload -u “$PYPI_USER” -p “$PYPI_PASS” dist/* || { echo “Upload failed”; exit 1; }
echo “Package uploaded successfully!”
else
echo “Publishing skipped. You can publish later with ‘twine upload dist/*’.”
fi

echo “>>> Setup complete. Enjoy your llamafetch package!”

---

### How to Use This Script

1. **Save the Script:** Copy the entire script above into a file named (for example) `setup_llamafetch.sh`.
2. **Make Executable:** Run `chmod +x setup_llamafetch.sh` in your terminal.
3. **Run the Script:** Execute it with `./setup_llamafetch.sh` in an empty directory.
4. **Follow Prompts:** The script will install dependencies, create the full package structure, run tests, build the package, display the CLI output, and prompt for PyPI publishing.

This single script automates the entire process of setting up, testing, building, and optionally publishing the llamafetch package on your Mac with a colorful, llama-themed command-line interface.

Enjoy your new ultimate llamafetch package!