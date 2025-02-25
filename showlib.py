import pkg_resources

# Danh sách các thư viện cần kiểm tra
libraries = [
    "python-dotenv",
    "langchain",
    "langchain-openai",
    "langchain-community",
    "langchain-core",
    "langchain-groq",
    "streamlit",
    "fastapi",
    "langserve",
    "pyngrok",
    "transformers",
    "torch",
    "numpy",
    "sounddevice",
    "scipy",
    "google-generativeai",
    "langchain_google_genai",
    "langchain_chroma",
    "chromadb"
]

# Kiểm tra phiên bản của từng thư viện
for lib in libraries:
    try:
        version = pkg_resources.get_distribution(lib).version
        print(f"{lib}=={version}")
    except pkg_resources.DistributionNotFound:
        print(f"{lib} is not installed")