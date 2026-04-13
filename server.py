#!/usr/bin/env python3
"""
CouncilGenius V8 - City of Melbourne
Production Server using http.server stdlib
"""

import os
import sys
import json
import csv
import hashlib
import re
import time
import logging
import datetime
import urllib.parse
import ipaddress
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import urllib.request

# Configuration
COUNCIL_NAME = "City of Melbourne"
COUNCIL_DOMAIN = "www.melbourne.vic.gov.au"
COUNCIL_PHONE = "+61 3 9658 9658"
MODEL = os.getenv("MODEL", "claude-sonnet-4-6")
MAX_TOKENS = 1024
PORT = int(os.getenv("PORT", 8080))
PROMPT_VERSION = "1.0"
BIN_LOOKUP_MODE = "none"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Knowledge base path
KB_PATH = Path(__file__).parent / "knowledge.txt"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
knowledge_base = ""
knowledge_hash = ""
knowledge_lines = 0
startup_time = time.time()


def load_knowledge_base():
    """Load and hash the knowledge base."""
    global knowledge_base, knowledge_hash, knowledge_lines

    if KB_PATH.exists():
        with open(KB_PATH, 'r', encoding='utf-8') as f:
            knowledge_base = f.read()

        knowledge_hash = hashlib.sha256(knowledge_base.encode()).hexdigest()
        knowledge_lines = len(knowledge_base.split('\n'))
        logger.info(f"Knowledge base loaded: {knowledge_lines} lines, hash: {knowledge_hash}")
    else:
        knowledge_base = ""
        knowledge_hash = hashlib.sha256(b"").hexdigest()
        knowledge_lines = 0
        logger.warning(f"Knowledge base not found at {KB_PATH}")


def filter_pii(text):
    """Filter personally identifiable information from text."""
    # Phone numbers
    text = re.sub(r'\b(?:\d{2}\s?)?\d{4}\s?\d{3}\s?\d{3}\b', '[PHONE]', text)
    text = re.sub(r'\b(?:\d{3}[-.]?)?\d{3}[-.]?\d{4}\b', '[PHONE]', text)

    # Email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)

    # Australian addresses (simplified)
    text = re.sub(r'\b(?:Street|Street|Road|Avenue|Drive|Lane|Court|Crescent|Close|Terrace|Place|Way|Parade)\b', '[ADDRESS]', text, flags=re.IGNORECASE)

    # Names (pattern: Capitalized words)
    text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', text)

    # Financial account numbers
    text = re.sub(r'\b\d{3}[-]?\d{3}[-]?\d{3}\b', '[ACCOUNT]', text)

    # Australian driver's license and ID patterns
    text = re.sub(r'\b[A-Z]{2}\d{6}\b', '[ID]', text)

    return text


def detect_australian_address(text):
    """Detect if text contains an Australian address."""
    # Australian postcode pattern (0000-9999)
    postcode_pattern = r'\b[0-9]{4}\b'

    # Australian states
    states = ['NSW', 'VIC', 'QLD', 'WA', 'SA', 'TAS', 'NT', 'ACT']

    has_postcode = bool(re.search(postcode_pattern, text))
    has_state = any(state in text.upper() for state in states)

    return has_postcode or has_state


def classify(text):
    """Classify question into categories (V9: logging only, never blocks queries)."""
    CATEGORIES = {
        'waste_bins': ['bin', 'collection', 'recycling', 'green waste', 'hard waste', 'rubbish', 'garbage', 'fogo', 'transfer station', 'tip', 'dump', 'landfill', 'waste', 'e-waste', 'e waste', 'mattress', 'furniture', 'skip'],
        'rates': ['rates', 'payment', 'concession', 'rebate', 'due date', 'valuation', 'rate notice', 'pay', 'overdue'],
        'planning': ['planning', 'permit', 'development', 'application', 'zone', 'heritage', 'overlay', 'building', 'construction', 'shed', 'extension', 'subdivision', 'build', 'construct', 'deck', 'fence', 'pool', 'renovation', 'carport', 'pergola', 'granny flat', 'demolish', 'retaining wall', 'setback', 'building permit', 'building work', 'dwelling'],
        'roads': ['pothole', 'road', 'maintenance', 'street', 'tree', 'street light', 'graffiti', 'footpath', 'report', 'issue', 'vandalism', 'damaged', 'broken', 'hazard', 'safety', 'drain', 'flooding', 'sign'],
        'parking': ['parking', 'fine', 'ticket', 'infringement', 'meter'],
        'pets': ['pet', 'dog', 'cat', 'animal', 'registration', 'microchip', 'barking', 'off-lead', 'off lead', 'dangerous dog', 'off-leash', 'off leash', 'desex', 'de-sex'],
        'property': ['property', 'land', 'address', 'search', 'valuation'],
        'family': ['kindergarten', 'kinder', 'childcare', 'maternal', 'immunisation', 'playgroup', 'family', 'children', 'youth', 'aged care', 'disability', 'seniors'],
        'community': ['library', 'libraries', 'pool', 'community', 'centre', 'center', 'program', 'recreation', 'swimming', 'sport', 'venue', 'hire', 'leisure', 'hall', 'tourism', 'visitor', 'museum', 'arts', 'culture', 'events', 'volunteer'],
        'food_business': ['food', 'business', 'registration', 'supplier', 'tender', 'procurement', 'vendor', 'restaurant', 'cafe'],
        'contact': ['phone', 'email', 'address', 'hours', 'contact', 'office', 'emergency', 'after hours', 'urgent', 'open', 'closed', 'when', 'where'],
        'environment': ['stormwater', 'contaminated', 'environment', 'septic', 'wastewater', 'climate', 'bushfire', 'fire', 'emergency', 'sustainability', 'solar', 'tree protection'],
        'legal': ['appeal', 'complaint', 'legal', 'ombudsman', 'foi', 'freedom of information', 'privacy', 'whistleblower'],
        'grants': ['grant', 'funding', 'support', 'community fund'],
        'local_laws': ['local law', 'bylaw', 'burning', 'burn off', 'camping', 'livestock', 'noise', 'fire', 'CFA', 'burn-off'],
        'forms': ['form', 'application', 'download', 'online', 'portal'],
        'potential_api_abuse': ['api', 'endpoint', 'json', 'curl', 'hack', 'inject', 'sql', 'script', 'exploit'],
        'off_topic': ['weather', 'football', 'recipe', 'joke', 'song'],
    }
    text_lower = text.lower()
    scores = {}
    for cat, keywords in CATEGORIES.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[cat] = score
    top = max(scores, key=scores.get) if max(scores.values()) > 0 else 'general'
    return top


def hash_ip(ip_address):
    """Hash IP address for privacy."""
    return hashlib.sha256(ip_address.encode()).hexdigest()[:16]


def log_query_basic(ip_address, filtered_question, response_time, category):
    """Log basic query information to JSONL."""
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "hashed_ip": hash_ip(ip_address),
        "filtered_question": filtered_question,
        "response_time_ms": response_time,
        "category": category
    }

    log_file = Path(__file__).parent / "query_log_basic.jsonl"
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry) + '\n')


def log_query_full(ip_address, filtered_question, response_time, category, filtered_answer, answer_length, thumbs, sources, follow_up):
    """Log full query information to JSONL."""
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "hashed_ip": hash_ip(ip_address),
        "filtered_question": filtered_question,
        "response_time_ms": response_time,
        "category": category,
        "filtered_answer": filtered_answer,
        "answer_length": answer_length,
        "thumbs": thumbs,
        "sources": sources,
        "follow_up": follow_up,
        "prompt_version": PROMPT_VERSION,
        "kb_hash": knowledge_hash
    }

    log_file = Path(__file__).parent / "query_log_full.jsonl"
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry) + '\n')


def log_feedback_csv(ip_address, question, answer, feedback, timestamp):
    """Log feedback to CSV."""
    csv_file = Path(__file__).parent / "feedback.csv"

    file_exists = csv_file.exists()

    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'hashed_ip', 'question', 'answer', 'feedback'])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'timestamp': timestamp,
            'hashed_ip': hash_ip(ip_address),
            'question': question,
            'answer': answer,
            'feedback': feedback
        })


def build_system_prompt(messages, bin_context=""):
    """Build system prompt from knowledge base with current date and version."""
    today = datetime.date.today().strftime('%A %d %B %Y')
    prompt = knowledge_base.replace('__CURRENT_DATE__', today)
    prompt = prompt.replace('__PROMPT_VERSION__', PROMPT_VERSION)
    if bin_context:
        prompt += f"\n\n--- LIVE BIN DATA ---\n{bin_context}"
    return prompt


def handle_search_protocol(question):
    """Parse search: protocol from knowledge base."""
    if question.startswith("search:"):
        search_term = question[7:].strip()

        # Parse URL directory format
        matches = []
        for line in knowledge_base.split('\n'):
            if search_term.lower() in line.lower():
                matches.append(line)

        return matches[:10] if matches else ["No results found for search term."]

    return None


class CouncilGeniusHandler(BaseHTTPRequestHandler):
    """HTTP request handler for CouncilGenius V8."""

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == '/':
            self.serve_page()
        elif path == '/health':
            self.serve_health()
        elif path == '/knowledge.txt':
            self.serve_knowledge()
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"Not Found\n")

    def do_POST(self):
        """Handle POST requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == '/chat':
            self.handle_chat()
        elif path == '/feedback':
            self.handle_feedback()
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"Not Found\n")

    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def serve_page(self):
        """Serve the main page."""
        page_path = Path(__file__).parent / "page.html"

        if page_path.exists():
            with open(page_path, 'r', encoding='utf-8') as f:
                content = f.read()

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(content.encode())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"Page not found\n")

    def serve_health(self):
        """Serve health check with V8 details."""
        uptime = time.time() - startup_time

        health = {
            "status": "healthy",
            "council": COUNCIL_NAME,
            "knowledge_loaded": knowledge_lines > 0,
            "knowledge_lines": knowledge_lines,
            "knowledge_hash": knowledge_hash,
            "bin_mode": BIN_LOOKUP_MODE,
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "uptime_seconds": int(uptime),
            "total_queries": self.count_queries()
        }

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(health).encode())

    def serve_knowledge(self):
        """Serve the knowledge base."""
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(knowledge_base.encode())

    def handle_chat(self):
        """Handle chat POST requests."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')

        try:
            data = json.loads(body)
            messages = data.get('messages', [])
            user_message = messages[-1]['content'] if messages else ''

            if not user_message:
                self.send_error_response(400, "Question required")
                return

            # Get client IP
            client_ip = self.client_address[0]

            # Check for search protocol
            search_results = handle_search_protocol(user_message)
            if search_results is not None:
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                resp = {
                    "response": "\n".join(search_results),
                    "category": "search"
                }
                self.wfile.write(json.dumps(resp).encode())
                return

            # Filter PII
            filtered_question = filter_pii(user_message)

            # Classify category (V9: logging only, never blocks queries)
            category = classify(user_message)

            # Build system prompt with bin context
            system_prompt = build_system_prompt(messages)

            # Call Anthropic API via urllib
            start_time = time.time()
            try:
                api_body = json.dumps({
                    'model': MODEL,
                    'max_tokens': MAX_TOKENS,
                    'system': system_prompt,
                    'messages': [{'role': m['role'], 'content': m['content']} for m in messages if isinstance(m, dict)]
                }).encode()

                req = urllib.request.Request(
                    'https://api.anthropic.com/v1/messages',
                    data=api_body,
                    headers={
                        'Content-Type': 'application/json',
                        'x-api-key': ANTHROPIC_API_KEY,
                        'anthropic-version': '2023-06-01'
                    }
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read())

                answer = result['content'][0]['text']
                response_time = int((time.time() - start_time) * 1000)

            except Exception as api_error:
                logger.error(f"API error: {str(api_error)}")
                self.send_json_response({
                    "response": f"Sorry, I couldn't process your question right now. Please try again, or call {COUNCIL_NAME} on {COUNCIL_PHONE} for help.",
                    "error": True,
                    "category": category
                })
                return

            # Filter answer
            filtered_answer = filter_pii(answer)

            # Log queries
            log_query_basic(client_ip, filtered_question, response_time, category)
            log_query_full(
                client_ip,
                filtered_question,
                response_time,
                category,
                filtered_answer,
                len(answer),
                None,
                [],
                False
            )

            # Send response
            self.send_json_response({
                "response": answer,
                "category": category,
                "bin_info": None
            })

        except json.JSONDecodeError:
            self.send_error_response(400, "Invalid JSON")
        except Exception as e:
            logger.error(f"Error in handle_chat: {str(e)}")
            self.send_error_response(500, "Internal server error")

    def handle_feedback(self):
        """Handle feedback POST requests."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')

        try:
            data = json.loads(body)
            question = data.get('question', '')
            answer = data.get('answer', '')
            feedback = data.get('feedback', '')

            client_ip = self.client_address[0]
            timestamp = datetime.datetime.utcnow().isoformat()

            log_feedback_csv(client_ip, question, answer, feedback, timestamp)

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success"}).encode())

        except Exception as e:
            logger.error(f"Error in handle_feedback: {str(e)}")
            self.send_error_response(500, "Internal server error")

    def send_json_response(self, data, status=200):
        """Send JSON response with CORS headers."""
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def send_error_response(self, code, message):
        """Send error response."""
        self.send_json_response({"error": message}, code)

    def count_queries(self):
        """Count total queries from logs."""
        log_file = Path(__file__).parent / "query_log_basic.jsonl"
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        return 0

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def main():
    """Start the server."""
    load_knowledge_base()

    print(f"\n{'='*60}")
    print(f"CouncilGenius V8 - {COUNCIL_NAME}")
    print(f"{'='*60}")
    print(f"Model: {MODEL}")
    print(f"Knowledge Lines: {knowledge_lines}")
    print(f"Knowledge Hash: {knowledge_hash}")
    print(f"Prompt Version: {PROMPT_VERSION}")
    print(f"Bin Mode: {BIN_LOOKUP_MODE}")
    print(f"Port: {PORT}")
    print(f"{'='*60}\n")

    server_address = ('', PORT)
    httpd = HTTPServer(server_address, CouncilGeniusHandler)

    logger.info(f"Starting server on port {PORT}")
    print(f"Server running at http://localhost:{PORT}/")
    print(f"Press Ctrl+C to stop\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        print("\nServer stopped.")
        sys.exit(0)


if __name__ == '__main__':
    main()
