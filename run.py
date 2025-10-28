"""
EGroupware RAG System - Main Entry Point
"""
import os


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from app.server import main  
if __name__ == '__main__':
    main()

