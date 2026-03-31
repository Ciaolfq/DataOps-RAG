# DataOps-RAG

DataOps-RAG is a Retrieval-Augmented Generation (RAG) system for data operations question answering. The project provides a complete pipeline from data loading to web-based interaction, leveraging ChromaDB for vector storage and Gradio for the user interface. Two versions of the pipeline are available: the main implementation in `scripts/` and an enhanced version in `scriptsv2/`.

## Directory Structure

```
dataops-rag/
├── .github/                     # GitHub configuration files
├── ASE2026_Reproducibility_Package/  # Reproducibility experiment materials
│   ├── dataops-qa/              # QA module
│   ├── chroma_db/               # ChromaDB vector database storage
│   ├── data/                    # Data files
│   ├── finding.txt              # Experimental findings or results
│   ├── gh.json                  # GitHub-related data or configuration
│   ├── sojson                   # Stack Overflow JSON data
│   └── scripts/                 # Main pipeline scripts
│       ├── .gradio/             # Gradio cache and configuration
│       ├── 01_load_data.py      # Step 1: Load and preprocess data
│       ├── 02_create_vector_db.py # Step 2: Create vector database
│       ├── 03_qa_system.py      # Step 3: QA system core logic
│       ├── 04_web_app.py        # Step 4: Web application interface
│       └── requirements.txt     # Python dependencies
├── scriptsv2/                   # Enhanced or alternative version
│   ├── 01_load_data.py          # Enhanced data loading
│   ├── 02_create_vector_db.py   # Enhanced vector DB creation
│   ├── 03_qa_system.py          # Enhanced QA logic
│   ├── 04_web_app.py            # Enhanced web interface
│   ├── python/                  # Additional Python modules
│   ├── README.md                # Documentation for scriptsv2
│   └── prompt.txt               # Prompt templates
└── README.md                    # Main project documentation
```

## Features

- **End-to-End RAG Pipeline**: Complete workflow from data loading to web deployment
- **Vector Database Integration**: Uses ChromaDB for efficient similarity search
- **Dual Implementation**: Two script versions for comparison or alternative approaches
- **Web Interface**: Built with Gradio for interactive user experience
- **Reproducibility Support**: Dedicated package with findings and experiment data
- **Prompt Engineering**: Centralized prompt templates for consistent LLM interactions
- **Multiple Data Sources**: Supports various data formats including GitHub and Stack Overflow data

## Pipeline Overview

The project follows a four-step pipeline available in two versions:

| Script | Description |
|--------|-------------|
| `01_load_data.py` | Load and preprocess raw data from the `data/` directory |
| `02_create_vector_db.py` | Generate embeddings and build ChromaDB vector store |
| `03_qa_system.py` | Implement RAG-based QA logic with retrieval and generation |
| `04_web_app.py` | Launch Gradio web application for user interaction |

### Version Comparison

| Version | Location | Description |
|---------|----------|-------------|
| Main | `ASE2026_Reproducibility_Package/scripts/` | Standard implementation used for reproducibility experiments |
| Enhanced | `scriptsv2/` | Improved version with potential optimizations, additional features, or alternative approaches |

Refer to `scriptsv2/README.md` for detailed information about the enhancements.

## Quick Start

### Prerequisites

- Python 3.9+
- pip package manager

### Installation

```bash
git clone https://github.com/your-repo/dataops-rag.git
cd dataops-rag/ASE2026_Reproducibility_Package
pip install -r scripts/requirements.txt
```

### Usage (Main Version)

Run the pipeline steps sequentially from the `scripts/` directory:

```bash
cd scripts

# Step 1: Load data
python 01_load_data.py

# Step 2: Create vector database
python 02_create_vector_db.py

# Step 3: Test QA system (optional)
python 03_qa_system.py --query "Your question here"

# Step 4: Launch web application
python 04_web_app.py
```

Once the web app is running, access the Gradio interface at `http://localhost:7860` to interact with the QA system.

### Usage (Enhanced Version)

To use the enhanced version:

```bash
cd scriptsv2

# Run enhanced pipeline
python 01_load_data.py
python 02_create_vector_db.py
python 03_qa_system.py
python 04_web_app.py
```

## Component Description

| Directory/File | Description |
|----------------|-------------|
| `ASE2026_Reproducibility_Package/dataops-qa/` | Main QA module with core functionality |
| `ASE2026_Reproducibility_Package/chroma_db/` | Persistent storage for document vectors and metadata |
| `ASE2026_Reproducibility_Package/data/` | Raw datasets or preprocessed documents |
| `ASE2026_Reproducibility_Package/finding.txt` | Experimental findings and results documentation |
| `ASE2026_Reproducibility_Package/gh.json` | GitHub data for experiments |
| `ASE2026_Reproducibility_Package/sojson` | Stack Overflow JSON data |
| `ASE2026_Reproducibility_Package/scripts/` | Main pipeline scripts with Gradio integration |
| `scriptsv2/` | Enhanced version of the pipeline with improvements |
| `scriptsv2/python/` | Additional Python modules for extended functionality |
| `scriptsv2/prompt.txt` | Prompt templates for LLM generation |

## Data Sources

The project supports multiple data sources for experimentation:

- **GitHub Data**: `gh.json` contains GitHub-related information for QA
- **Stack Overflow Data**: `sojson` contains Stack Overflow Q&A pairs
- **Custom Data**: Add your own documents to the `data/` directory

## Customization

### Adding New Data

1. Place your documents in the `data/` directory or update existing JSON files
2. Modify `01_load_data.py` if custom preprocessing is required
3. Re-run the pipeline steps

### Modifying Prompts

Edit `prompt.txt` (in either `scripts/` or `scriptsv2/`) to customize LLM prompt templates.

### Switching Between Versions

- Use the main version for reproducibility experiments
- Use `scriptsv2/` for enhanced features and testing alternative approaches

### Changing Embedding Model

Modify the embedding model configuration in the `02_create_vector_db.py` files.

## Experimental Findings

The `finding.txt` file contains documentation of experimental results, observations, and insights from running the RAG system with different configurations and data sources.

## Contributing

Contributions are welcome! Please ensure:

- Code follows Python best practices
- New features are documented
- Pipeline scripts maintain backward compatibility
- Updates to `scriptsv2/` are explained in its README
- Experimental findings are documented appropriately

## License

This project is licensed under the [MIT License](LICENSE).

---

Let me know if you'd like me to add more details about the differences between the two script versions or expand on any other section!
