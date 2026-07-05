import logging
import os
from datetime import datetime

def setup_logging(log_dir='outputs/logs'):
    """Setup logging configuration."""
    
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def create_directories():
    """Create necessary project directories."""
    
    dirs = ['data', 'models', 'outputs', 'outputs/logs', 'outputs/plots', 'app']
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("Project directories created.")

def format_currency(value):
    """Format number as currency."""
    return f"${value:,.2f}"

def format_percentage(value):
    """Format number as percentage."""
    return f"{value:.2f}%"
