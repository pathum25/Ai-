"""
Flask API for Multi-Model Hugging Face AI System
Provides RESTful endpoints for various AI tasks
"""

from flask import Flask, request, jsonify
from model_manager import ModelManager
import logging
from typing import Dict, Any
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize Model Manager
manager = ModelManager()


# ==================== Health & Info Endpoints ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "available_models": manager.list_models(),
        "cache_info": manager.get_cache_info()
    }), 200


@app.route('/models', methods=['GET'])
def list_models():
    """List all available models"""
    models = manager.list_models()
    model_details = {name: manager.get_model_info(name) for name in models}
    
    return jsonify({
        "total_models": len(models),
        "models": model_details
    }), 200


@app.route('/models/<model_name>', methods=['GET'])
def get_model_details(model_name: str):
    """Get details about a specific model"""
    info = manager.get_model_info(model_name)
    
    if not info:
        return jsonify({"error": f"Model '{model_name}' not found"}), 404
    
    return jsonify({
        "model": model_name,
        "details": info,
        "cached": model_name in manager.pipelines_cache
    }), 200


# ==================== Text Generation Endpoints ====================

@app.route('/api/generate', methods=['POST'])
def text_generation():
    """Generate text using GPT-2 or other text generation models"""
    try:
        data = request.json
        
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' field"}), 400
        
        prompt = data.get('prompt')
        model_name = data.get('model', 'text_generation')
        max_length = data.get('max_length', 100)
        
        result = manager.infer_text_generation(model_name, prompt, max_length)
        
        if result is None:
            return jsonify({"error": f"Failed to generate text with model '{model_name}'"}), 500
        
        return jsonify({
            "prompt": prompt,
            "model": model_name,
            "generated_text": result
        }), 200
    
    except Exception as e:
        logger.error(f"Error in text generation: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ==================== Text Classification Endpoints ====================

@app.route('/api/classify', methods=['POST'])
def text_classification():
    """Classify text (sentiment analysis, etc.)"""
    try:
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        text = data.get('text')
        model_name = data.get('model', 'text_classification')
        
        result = manager.infer_text_classification(model_name, text)
        
        if result is None:
            return jsonify({"error": f"Failed to classify with model '{model_name}'"}), 500
        
        return jsonify({
            "text": text,
            "model": model_name,
            "classification": result
        }), 200
    
    except Exception as e:
        logger.error(f"Error in text classification: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ==================== Named Entity Recognition Endpoints ====================

@app.route('/api/ner', methods=['POST'])
def named_entity_recognition():
    """Extract named entities from text"""
    try:
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        text = data.get('text')
        model_name = data.get('model', 'named_entity_recognition')
        
        result = manager.infer_named_entity_recognition(model_name, text)
        
        if result is None:
            return jsonify({"error": f"Failed to extract entities with model '{model_name}'"}), 500
        
        return jsonify({
            "text": text,
            "model": model_name,
            "entities": result
        }), 200
    
    except Exception as e:
        logger.error(f"Error in NER: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ==================== Question Answering Endpoints ====================

@app.route('/api/qa', methods=['POST'])
def question_answering():
    """Answer questions based on context"""
    try:
        data = request.json
        
        if not data or 'question' not in data or 'context' not in data:
            return jsonify({"error": "Missing 'question' or 'context' field"}), 400
        
        question = data.get('question')
        context = data.get('context')
        model_name = data.get('model', 'question_answering')
        
        result = manager.infer_question_answering(model_name, question, context)
        
        if result is None:
            return jsonify({"error": f"Failed to answer with model '{model_name}'"}), 500
        
        return jsonify({
            "question": question,
            "context": context[:200] + "..." if len(context) > 200 else context,
            "model": model_name,
            "answer": result
        }), 200
    
    except Exception as e:
        logger.error(f"Error in QA: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ==================== Summarization Endpoints ====================

@app.route('/api/summarize', methods=['POST'])
def summarization():
    """Summarize text"""
    try:
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        text = data.get('text')
        model_name = data.get('model', 'text_summarization')
        max_length = data.get('max_length', 130)
        min_length = data.get('min_length', 30)
        
        result = manager.infer_summarization(model_name, text, max_length, min_length)
        
        if result is None:
            return jsonify({"error": f"Failed to summarize with model '{model_name}'"}), 500
        
        return jsonify({
            "original_length": len(text),
            "model": model_name,
            "summary": result,
            "summary_length": len(result)
        }), 200
    
    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ==================== Zero-Shot Classification Endpoints ====================

@app.route('/api/zero-shot', methods=['POST'])
def zero_shot_classification():
    """Zero-shot classification"""
    try:
        data = request.json
        
        if not data or 'text' not in data or 'labels' not in data:
            return jsonify({"error": "Missing 'text' or 'labels' field"}), 400
        
        text = data.get('text')
        labels = data.get('labels')
        model_name = data.get('model', 'zero_shot_classification')
        
        if not isinstance(labels, list):
            return jsonify({"error": "'labels' must be a list"}), 400
        
        result = manager.infer_zero_shot(model_name, text, labels)
        
        if result is None:
            return jsonify({"error": f"Failed with model '{model_name}'"}), 500
        
        return jsonify({
            "text": text,
            "labels": labels,
            "model": model_name,
            "classification": result
        }), 200
    
    except Exception as e:
        logger.error(f"Error in zero-shot: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ==================== Batch Processing Endpoints ====================

@app.route('/api/batch', methods=['POST'])
def batch_processing():
    """Process multiple inputs in batch"""
    try:
        data = request.json
        
        if not data or 'inputs' not in data or 'task' not in data:
            return jsonify({"error": "Missing 'inputs', 'task' field"}), 400
        
        inputs = data.get('inputs')
        task = data.get('task')
        model_name = data.get('model', 'text_generation')
        
        if not isinstance(inputs, list):
            return jsonify({"error": "'inputs' must be a list"}), 400
        
        results = manager.batch_inference(model_name, task, inputs)
        
        return jsonify({
            "task": task,
            "model": model_name,
            "total_inputs": len(inputs),
            "results": results
        }), 200
    
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ==================== Cache Management Endpoints ====================

@app.route('/cache', methods=['GET'])
def get_cache_info():
    """Get cache information"""
    return jsonify({
        "cache_info": manager.get_cache_info()
    }), 200


@app.route('/cache', methods=['DELETE'])
def clear_cache():
    """Clear the model cache"""
    manager.clear_cache()
    return jsonify({
        "message": "Cache cleared successfully",
        "cache_info": manager.get_cache_info()
    }), 200


# ==================== Error Handlers ====================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500


# ==================== Main ====================

if __name__ == '__main__':
    logger.info("Starting Multi-Model Hugging Face AI API")
    logger.info(f"Available models: {manager.list_models()}")
    app.run(debug=True, host='0.0.0.0', port=5000)
