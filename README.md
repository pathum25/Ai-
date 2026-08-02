# Advanced Multi-Model Hugging Face AI System 🚀

A sophisticated, production-ready AI system leveraging multiple Hugging Face transformer models for various NLP tasks.

## Features ✨

- **Multiple AI Models**: 6+ pre-configured models for different tasks
- **Task Support**:
  - 📝 Text Generation (GPT-2)
  - 😊 Sentiment Analysis & Classification (DistilBERT)
  - 🏷️ Named Entity Recognition (BERT-NER)
  - ❓ Question Answering (DistilBERT-QA)
  - 📄 Text Summarization (T5)
  - 🎯 Zero-Shot Classification (BART)

- **Advanced Features**:
  - Intelligent model caching
  - Batch processing support
  - GPU/CPU optimization
  - RESTful API with Flask
  - Comprehensive error handling
  - Logging and monitoring

## Installation 📦

```bash
# Clone repository
git clone https://github.com/pathum25/Ai-.git
cd Ai-

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
```

## Quick Start 🎯

### 1. Start the API Server

```bash
python app.py
```

The API will be available at `http://localhost:5000`

### 2. Use the Python Manager Directly

```python
from model_manager import ModelManager

# Initialize manager
manager = ModelManager()

# List available models
print(manager.list_models())

# Text Generation
result = manager.infer_text_generation(
    'text_generation',
    'Once upon a time',
    max_length=100
)
print(result)

# Sentiment Analysis
result = manager.infer_text_classification(
    'text_classification',
    'I love this product!'
)
print(result)

# Named Entity Recognition
result = manager.infer_named_entity_recognition(
    'named_entity_recognition',
    'John works at Google in Mountain View'
)
print(result)
```

## API Endpoints 🔌

### Health & Info
- `GET /health` - Health check
- `GET /models` - List all models
- `GET /models/<model_name>` - Get model details

### Text Generation
- `POST /api/generate` - Generate text

**Request**:
```json
{
  "prompt": "Once upon a time",
  "model": "text_generation",
  "max_length": 100
}
```

### Text Classification
- `POST /api/classify` - Classify text

**Request**:
```json
{
  "text": "I love this product!",
  "model": "text_classification"
}
```

### Named Entity Recognition
- `POST /api/ner` - Extract entities

**Request**:
```json
{
  "text": "John works at Google",
  "model": "named_entity_recognition"
}
```

### Question Answering
- `POST /api/qa` - Answer questions

**Request**:
```json
{
  "question": "Who works at Google?",
  "context": "John works at Google",
  "model": "question_answering"
}
```

### Text Summarization
- `POST /api/summarize` - Summarize text

**Request**:
```json
{
  "text": "Long text to summarize...",
  "model": "text_summarization",
  "max_length": 130,
  "min_length": 30
}
```

### Zero-Shot Classification
- `POST /api/zero-shot` - Zero-shot classify

**Request**:
```json
{
  "text": "This is about politics",
  "labels": ["sports", "politics", "technology"],
  "model": "zero_shot_classification"
}
```

### Batch Processing
- `POST /api/batch` - Process multiple inputs

**Request**:
```json
{
  "inputs": ["Input 1", "Input 2", "Input 3"],
  "task": "text-classification",
  "model": "text_classification"
}
```

### Cache Management
- `GET /cache` - Get cache info
- `DELETE /cache` - Clear cache

## Configuration 📋

Edit `config.json` to customize models:

```json
{
  "models": {
    "your_model": {
      "model_id": "huggingface-model-id",
      "task": "task-type",
      "description": "Model description"
    }
  },
  "settings": {
    "cache_dir": "./model_cache",
    "device": "cpu",
    "enable_gpu": false
  }
}
```

## Performance Optimization ⚡

### Enable GPU Support
```json
{
  "settings": {
    "enable_gpu": true
  }
}
```

### Adjust Batch Size
```json
{
  "settings": {
    "batch_size": 64
  }
}
```

## Model Details 📚

| Model | Task | Model ID | Speed | Accuracy |
|-------|------|----------|-------|----------|
| GPT-2 | Text Generation | `openai-community/gpt2` | ⚡⚡ | ⭐⭐⭐⭐ |
| DistilBERT | Classification | `distilbert-base-uncased-finetuned-sst-2-english` | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| BERT-NER | NER | `dslim/bert-base-uncased-finetuned-ner` | ⚡⚡ | ⭐⭐⭐⭐⭐ |
| T5 | Summarization | `google-t5/t5-small` | ⚡⚡ | ⭐⭐⭐⭐ |
| BART | Zero-Shot | `facebook/bart-large-mnli` | ⚡ | ⭐⭐⭐⭐⭐ |

## Example Usage

```bash
# Generate text
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "AI is", "max_length": 50}'

# Classify sentiment
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'

# Extract entities
curl -X POST http://localhost:5000/api/ner \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple Inc. was founded by Steve Jobs."}'

# Summarize text
curl -X POST http://localhost:5000/api/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Long text here..."}'
```

## Architecture 🏗️

```
Ai-/
├── app.py                 # Flask API endpoints
├── model_manager.py       # Sophisticated model manager
├── config.json           # Model configuration
├── requirements.txt      # Python dependencies
├── .env.example         # Environment template
└── README.md            # Documentation
```

## Logging 📊

All operations are logged. Check logs for debugging:

```python
import logging
logger = logging.getLogger(__name__)
logger.info("Your log message")
```

## Performance Tips 🚀

1. **Use GPU** for faster inference on large texts
2. **Enable caching** to avoid re-downloading models
3. **Batch processing** for multiple inputs
4. **Use lightweight models** (DistilBERT) for real-time applications
5. **Clear cache** when switching models frequently

## Troubleshooting 🔧

### Out of Memory
- Reduce `batch_size` in config.json
- Use lighter models (DistilBERT instead of BERT)
- Enable GPU if available

### Slow Inference
- Enable GPU support
- Increase batch size
- Use smaller max_length values

### Model Not Found
- Check Hugging Face model ID in config.json
- Ensure internet connection for first download
- Verify Hugging Face hub is accessible

## Contributing 🤝

Contributions welcome! Feel free to:
- Add new models
- Improve performance
- Add new task types
- Report issues

## License 📄

MIT License - Feel free to use in your projects!

## Support 💬

For issues and questions, open a GitHub issue or contact the maintainers.

---

**Made with ❤️ for AI enthusiasts**
