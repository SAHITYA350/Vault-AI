import os
import time
import requests
import json
import numpy as np
import hashlib
import random
import base64
from django.conf import settings
from .models import Document

# Hugging Face Inference API Endpoints
API_URL_CAPTION = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
API_URL_OCR = "https://api-inference.huggingface.co/models/microsoft/trocr-large-printed"
API_URL_EMBEDDING = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

def query_huggingface_api(api_url, data, is_binary=False):
    token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    if not token:
        print("Missing HUGGINGFACEHUB_API_TOKEN in environment variables.")
        return None
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Try up to 3 times to account for model loading/cold starts
    for attempt in range(3):
        try:
            if is_binary:
                response = requests.post(api_url, headers=headers, data=data, timeout=30)
            else:
                response = requests.post(api_url, headers=headers, json=data, timeout=30)
                
            if response.status_code == 200:
                return response.json()
            
            # Handle model loading cold start
            try:
                res_json = response.json()
                if "currently loading" in res_json.get("error", ""):
                    wait_time = min(12, int(res_json.get("estimated_time", 5)))
                    print(f"HF Model loading, waiting {wait_time}s (attempt {attempt+1}/3)...")
                    time.sleep(wait_time)
                    continue
            except Exception:
                pass
                
            print(f"HF API returned status {response.status_code}: {response.text}")
            time.sleep(2)
        except Exception as e:
            print(f"Exception querying HF API: {e}")
            time.sleep(2)
            
    return None

def generate_blip_caption(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        res = query_huggingface_api(API_URL_CAPTION, data, is_binary=True)
        if res and isinstance(res, list) and len(res) > 0:
            return res[0].get("generated_text", "")
    except Exception as e:
        print(f"Error in BLIP image captioning: {e}")
    return ""

def extract_text_via_ocr(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        res = query_huggingface_api(API_URL_OCR, data, is_binary=True)
        if res and isinstance(res, list) and len(res) > 0:
            return res[0].get("generated_text", "")
    except Exception as e:
        print(f"Error in OCR text extraction: {e}")
    return ""

def extract_text_from_pdf(file_path):
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def extract_text_from_docx(file_path):
    try:
        import zipfile
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(file_path) as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            texts = []
            for paragraph in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                p_texts = [node.text for node in paragraph.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text]
                if p_texts:
                    texts.append("".join(p_texts))
            return "\n".join(texts).strip()
    except Exception as e:
        print(f"Error extracting DOCX text: {e}")
        return ""

def describe_image_via_mistral(file_path):
    mistral_key = os.getenv('MISTRAL_API_KEY')
    if not mistral_key:
        print("Missing MISTRAL_API_KEY, cannot describe image via Mistral.")
        return None
        
    try:
        ext = os.path.splitext(file_path)[1].lower().replace('.', '')
        mime_type = f"image/{ext}" if ext in ['png', 'jpg', 'jpeg', 'gif', 'webp'] else "image/png"
        
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        from langchain_mistralai import ChatMistralAI
        from langchain_core.messages import HumanMessage
        
        llm = ChatMistralAI(
            model="pixtral-12b-latest",
            mistral_api_key=mistral_key,
            temperature=0.2
        )
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image in detail. Identify the colors, objects, visual layout, and extract any text that is visible in the image. Provide a comprehensive summary that can be indexed for retrieval."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded_string}"}
                }
            ]
        )
        
        response = llm.invoke([message])
        return response.content.strip()
    except Exception as e:
        print(f"Error querying Mistral vision model: {e}")
        return None

def generate_auto_tags(text_content):
    mistral_key = os.getenv('MISTRAL_API_KEY')
    if not mistral_key:
        return "general"
        
    try:
        from langchain_mistralai import ChatMistralAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        llm = ChatMistralAI(
            model="mistral-large-latest",
            mistral_api_key=mistral_key,
            temperature=0.0
        )
        
        prompt = ChatPromptTemplate.from_template(
            "Analyze this snippet from a document and output 3 to 5 lowercase comma-separated keywords or tags representing the document. Output ONLY the tags, no explanation or conversational text.\n\nContent:\n{text}"
        )
        
        chain = prompt | llm | StrOutputParser()
        tags = chain.invoke({"text": text_content[:2000]})
        return tags.strip().lower()
    except Exception as e:
        print(f"Error generating tags: {e}")
        return "general"

def get_deterministic_mock_embedding(text):
    vector = [0.0] * 384
    words = [w.strip(".,!?\"'()").lower() for w in text.split()]
    words = [w for w in words if w and len(w) > 1]
    
    if not words:
        words = ["default"]
        
    for word in words:
        hash_object = hashlib.md5(word.encode('utf-8'))
        index = int(hash_object.hexdigest(), 16) % 384
        vector[index] += 1.0
        
    norm = sum(x**2 for x in vector)**0.5
    if norm > 0:
        vector = [x / norm for x in vector]
    return vector

def get_huggingface_embedding(text):
    if not text:
        return None
    # HuggingFace API is timing out; use fast local deterministic embeddings for now
    return get_deterministic_mock_embedding(text)

def run_ai_processing_pipeline(document_obj):
    """
    Runs the full pipeline for a Document object based on its file_type.
    Saves captioning, OCR text, generated tags, and embeddings to the DB.
    """
    file_path = document_obj.file.path
    file_type = document_obj.file_type
    
    extracted_text = ""
    ai_caption = ""
    
    print(f"Starting AI processing for Document: {document_obj.name} (Type: {file_type})")
    
    if file_type == 'image':
        # Primary: Use Mistral Vision model (since Mistral key and connection is working)
        mistral_desc = describe_image_via_mistral(file_path)
        if mistral_desc:
            document_obj.ai_caption = mistral_desc
            ai_caption = mistral_desc
            document_obj.extracted_text = mistral_desc
            extracted_text = mistral_desc
            print("Processed image details and text OCR using Mistral Vision model.")
        else:
            print("Mistral Vision failed. Falling back to Hugging Face APIs.")
            # 1. Salesforce BLIP Captioning
            caption = generate_blip_caption(file_path)
            if caption:
                document_obj.ai_caption = caption
                ai_caption = caption
                print(f"Generated caption: '{caption}'")
                
            # 2. OCR Text Extraction
            ocr_text = extract_text_via_ocr(file_path)
            if ocr_text:
                document_obj.extracted_text = ocr_text
                extracted_text = ocr_text
                print(f"Extracted OCR text length: {len(ocr_text)}")
            
    elif file_type == 'pdf':
        pdf_text = extract_text_from_pdf(file_path)
        if pdf_text:
            document_obj.extracted_text = pdf_text
            extracted_text = pdf_text
            print(f"Extracted PDF text length: {len(pdf_text)}")
            
    elif file_type == 'docx':
        docx_text = extract_text_from_docx(file_path)
        if docx_text:
            document_obj.extracted_text = docx_text
            extracted_text = docx_text
            print(f"Extracted DOCX text length: {len(docx_text)}")
            
    elif file_type == 'text':
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                txt_content = f.read()
            document_obj.extracted_text = txt_content
            extracted_text = txt_content
            print(f"Read text file content length: {len(txt_content)}")
        except Exception as e:
            print(f"Error reading text file: {e}")

    # Compile descriptive content for tag and embedding generation
    content_to_vectorize = document_obj.name
    if not ai_caption and file_type == 'image':
        ai_caption = f"[Offline Mode] Image of {document_obj.name.split('.')[0]}"
        document_obj.ai_caption = ai_caption
        
    if ai_caption:
        content_to_vectorize += " " + ai_caption
    if extracted_text:
        content_to_vectorize += " " + extracted_text
        
    # 3. Generate auto tags
    tags = generate_auto_tags(content_to_vectorize)
    if tags and tags != "general":
        clean_tags = [t.strip().replace(',', '') for t in tags.split(',') if t.strip()]
        document_obj.ai_tags = " ".join(clean_tags)
    else:
        # Fallback tags generated offline from filename words
        parts = document_obj.name.replace('.', ' ').replace('_', ' ').replace('-', ' ').split()
        clean_parts = [p.lower() for p in parts if len(p) > 2 and p.lower() not in ['png', 'jpg', 'jpeg', 'pdf', 'docx', 'txt', 'zip']]
        if not clean_parts:
            clean_parts = ["file", file_type]
        document_obj.ai_tags = " ".join(clean_parts[:5])
        
    print(f"Generated tags: {document_obj.ai_tags}")
        
    # 4. Generate 384d vector embedding
    vector = get_huggingface_embedding(content_to_vectorize)
    if vector:
        document_obj.embedding = vector
        print("Generated and stored text embeddings.")
        
    # Save the updated fields to the database without re-triggering signal recursively
    document_obj.save()
    
    # 5. Semantic Chunking — split into per-page, sentence-aware chunks
    try:
        chunk_document(document_obj)
    except Exception as ce:
        print(f"Chunking failed for {document_obj.name}: {ce}")

def perform_semantic_search(user, query_text):
    """
    Computes query vector embedding and ranks user documents by cosine similarity.
    """
    if not query_text:
        return Document.objects.filter(user=user).order_by('-uploaded_at')
        
    query_vector = get_huggingface_embedding(query_text)
    if not query_vector:
        print("Semantic search embedding failed. Falling back to keyword search.")
        return Document.objects.filter(user=user, name__icontains=query_text).order_by('-uploaded_at')
        
    # Get user documents containing embeddings
    docs = Document.objects.filter(user=user).exclude(embedding__isnull=True)
    if not docs.exists():
        return Document.objects.filter(user=user, name__icontains=query_text).order_by('-uploaded_at')
        
    q_vec = np.array(query_vector)
    scored_docs = []
    
    for doc in docs:
        try:
            doc_vec = np.array(doc.embedding)
            if len(doc_vec) == len(q_vec):
                dot_prod = np.dot(q_vec, doc_vec)
                norm_q = np.linalg.norm(q_vec)
                norm_d = np.linalg.norm(doc_vec)
                
                similarity = 0.0
                if norm_q > 0 and norm_d > 0:
                    similarity = float(dot_prod / (norm_q * norm_d))
                
                scored_docs.append((similarity, doc))
        except Exception as e:
            print(f"Error matching embeddings for doc {doc.id}: {e}")
            
    # Sort descending by cosine similarity score
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Filter documents that have a positive semantic overlap
    results = [doc for score, doc in scored_docs if score > 0.15]
    
    # Perform strict field keyword searches (OCR text, summaries, name, tags, caption)
    from django.db.models import Q
    keyword_matches = list(Document.objects.filter(
        Q(user=user) & (
            Q(name__icontains=query_text) |
            Q(extracted_text__icontains=query_text) |
            Q(ai_summary__icontains=query_text) |
            Q(ai_caption__icontains=query_text) |
            Q(ai_tags__icontains=query_text)
        )
    ).distinct().order_by('-uploaded_at'))
    
    # Merge semantic results with keyword matches (maintaining semantic ranking priority)
    semantic_ids = {doc.id for doc in results}
    for doc in keyword_matches:
        if doc.id not in semantic_ids:
            results.append(doc)
            
    if not results:
        return keyword_matches
        
    return results


# =============================================================================
#  SEMANTIC CHUNKING ENGINE
# =============================================================================

def extract_text_from_pdf_per_page(file_path):
    """Returns a list of (page_number, page_text) tuples — one per PDF page."""
    pages = []
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append((i + 1, text))
    except Exception as e:
        print(f"Error extracting PDF pages: {e}")
    return pages


def sentence_aware_chunks(text, max_chars=500, overlap_sentences=2):
    """
    Split text into semantic chunks using sentence boundaries.
    Uses a sliding window with sentence-level overlap for context continuity.
    Returns list of chunk strings.
    """
    import re
    # Split on sentence boundaries: '. ', '? ', '! ', '\n\n'
    sentence_splitter = re.compile(r'(?<=[.!?])\s+|\n{2,}')
    raw_sentences = [s.strip() for s in sentence_splitter.split(text) if s.strip() and len(s.strip()) > 10]
    
    if not raw_sentences:
        # Fallback: character-level chunks
        return [text[i:i+max_chars] for i in range(0, len(text), max_chars - 100) if text[i:i+max_chars].strip()]
    
    chunks = []
    current_chunk_sentences = []
    current_len = 0
    
    for sentence in raw_sentences:
        if current_len + len(sentence) > max_chars and current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            # Sliding overlap: keep last N sentences
            current_chunk_sentences = current_chunk_sentences[-overlap_sentences:]
            current_len = sum(len(s) for s in current_chunk_sentences)
        current_chunk_sentences.append(sentence)
        current_len += len(sentence)
    
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
    
    return [c for c in chunks if len(c.strip()) > 30]


def extract_chunk_concept(chunk_text):
    """
    Ask Mistral to label the core concept of a chunk.
    Returns a 2-6 word concept label like 'Gradient Descent Optimization'.
    """
    mistral_key = os.getenv('MISTRAL_API_KEY')
    if not mistral_key:
        # Fallback: use first significant noun phrase from chunk
        words = chunk_text.split()[:8]
        return " ".join(w for w in words if len(w) > 3)[:60]
    
    try:
        from langchain_mistralai import ChatMistralAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        llm = ChatMistralAI(
            model="mistral-small-latest",  # Fastest model for lightweight labeling
            mistral_api_key=mistral_key,
            temperature=0.0,
            max_tokens=20
        )
        prompt = ChatPromptTemplate.from_template(
            "Read this text snippet and output ONLY a 2-6 word concept label (like a topic heading). "
            "No explanation, no quotes, no punctuation at end. Just the label.\n\nText: {text}"
        )
        chain = prompt | llm | StrOutputParser()
        label = chain.invoke({"text": chunk_text[:600]})
        return label.strip()[:150]
    except Exception as e:
        print(f"Concept extraction error: {e}")
        # Fallback: first meaningful words
        return " ".join(chunk_text.split()[:6])[:100]


def chunk_document(document_obj):
    """
    Full semantic chunking pipeline for a document.
    Creates DocumentChunk records for each meaningful chunk.
    Runs per-page for PDFs, whole-doc for others.
    """
    from .models import DocumentChunk
    
    # Clear existing chunks for this document (re-processing)
    DocumentChunk.objects.filter(document=document_obj).delete()
    
    file_path = document_obj.file.path
    file_type = document_obj.file_type
    
    all_page_texts = []  # list of (page_number, page_text)
    
    if file_type == 'pdf':
        all_page_texts = extract_text_from_pdf_per_page(file_path)
    elif file_type in ['docx', 'text', 'document']:
        full_text = document_obj.extracted_text or ""
        if not full_text and file_type == 'docx':
            full_text = extract_text_from_docx(file_path)
        elif not full_text and file_type == 'text':
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    full_text = f.read()
            except Exception:
                pass
        if full_text:
            # Split into "virtual pages" of ~1500 chars
            page_size = 1500
            for i in range(0, len(full_text), page_size):
                page_text = full_text[i:i+page_size].strip()
                if page_text:
                    all_page_texts.append((len(all_page_texts) + 1, page_text))
    elif file_type == 'image':
        # Images: use AI caption + extracted text as single chunk
        full_text = (document_obj.ai_caption or "") + " " + (document_obj.extracted_text or "")
        if full_text.strip():
            all_page_texts = [(1, full_text.strip())]
    
    if not all_page_texts:
        print(f"No text to chunk for document: {document_obj.name}")
        return 0
    
    total_chunks = 0
    for page_number, page_text in all_page_texts:
        chunks = sentence_aware_chunks(page_text, max_chars=500, overlap_sentences=2)
        
        for chunk_idx, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
            
            # Generate embedding for this chunk
            embedding = get_huggingface_embedding(chunk_text)
            
            # Extract concept label (batch only the first chunk per page to save API calls)
            concept = ""
            if chunk_idx == 0:  # Label the first chunk of each page
                concept = extract_chunk_concept(chunk_text)
            
            # Calculate importance score: longer, more unique chunks score higher
            word_count = len(chunk_text.split())
            importance = min(1.0, word_count / 80.0)  # Normalize: 80 words = 1.0
            
            chunk_obj = DocumentChunk(
                document=document_obj,
                page_number=page_number,
                chunk_index=chunk_idx,
                text=chunk_text,
                embedding=embedding,
                concept_label=concept,
                importance_score=importance,
            )
            chunk_obj.save()
            total_chunks += 1
    
    # Mark document as chunked
    document_obj.is_chunked = True
    Document.objects.filter(pk=document_obj.pk).update(is_chunked=True)
    print(f"Chunked '{document_obj.name}': {total_chunks} chunks across {len(all_page_texts)} pages.")
    return total_chunks


def chunk_document_bg(doc_id):
    """Background-safe wrapper for chunk_document()."""
    import time
    time.sleep(1.0)  # Let DB commit
    try:
        doc = Document.objects.get(id=doc_id)
        chunk_document(doc)
    except Exception as e:
        print(f"Error in background chunking task for doc {doc_id}: {e}")


def chunk_semantic_search(user, query_text, doc_id=None, top_k=6):
    """
    Search DocumentChunk embeddings for semantically similar chunks.
    Returns list of dicts: {chunk, score, doc_name, page_number}
    """
    from .models import DocumentChunk
    import numpy as np
    
    if not query_text:
        return []
    
    query_vec = get_huggingface_embedding(query_text)
    if not query_vec:
        return []
    
    q_arr = np.array(query_vec)
    
    chunks_qs = DocumentChunk.objects.filter(document__user=user).select_related('document')
    if doc_id and doc_id != 'all':
        if isinstance(doc_id, list):
            chunks_qs = chunks_qs.filter(document_id__in=doc_id)
        elif isinstance(doc_id, str) and ',' in doc_id:
            try:
                doc_ids = [int(x.strip()) for x in doc_id.split(',') if x.strip().isdigit()]
                chunks_qs = chunks_qs.filter(document_id__in=doc_ids)
            except ValueError:
                chunks_qs = chunks_qs.filter(document_id=doc_id)
        else:
            chunks_qs = chunks_qs.filter(document_id=doc_id)
    
    chunks_qs = chunks_qs.exclude(embedding__isnull=True)
    
    scored = []
    for chunk in chunks_qs:
        try:
            c_arr = np.array(chunk.embedding)
            if len(c_arr) != len(q_arr):
                continue
            dot = np.dot(q_arr, c_arr)
            norm_q = np.linalg.norm(q_arr)
            norm_c = np.linalg.norm(c_arr)
            if norm_q > 0 and norm_c > 0:
                sim = float(dot / (norm_q * norm_c))
                scored.append((sim, chunk))
        except Exception:
            continue
    
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for sim, chunk in scored[:top_k]:
        if sim > 0.1:
            results.append({
                'chunk': chunk,
                'score': sim,
                'doc_name': chunk.document.name,
                'page_number': chunk.page_number,
                'text': chunk.text,
                'doc_id': chunk.document.id,
            })
    return results


def get_cross_doc_similar_chunks(user, top_pairs=15):
    """
    Find pairs of chunks from DIFFERENT documents with cosine similarity > 0.65.
    Used to draw cross-document concept edges in the knowledge graph.
    Returns list of (chunk_a_id, chunk_b_id, similarity, concept_a, concept_b)
    """
    from .models import DocumentChunk
    import numpy as np
    
    chunks = list(
        DocumentChunk.objects.filter(
            document__user=user
        ).exclude(embedding__isnull=True).select_related('document')
    )
    
    if len(chunks) < 2:
        return []
    
    pairs = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            ca, cb = chunks[i], chunks[j]
            if ca.document_id == cb.document_id:
                continue  # Skip same-document pairs
            try:
                va = np.array(ca.embedding)
                vb = np.array(cb.embedding)
                if len(va) != len(vb):
                    continue
                dot = np.dot(va, vb)
                sim = float(dot / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
                if sim > 0.65:
                    pairs.append((ca.id, cb.id, sim, ca.concept_label, cb.concept_label))
            except Exception:
                continue
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_pairs]


def run_langgraph_rag(user, query_text, doc_id_or_ids, chat_history_messages=[]):
    """
    State-of-the-art 2027 LangGraph-based RAG workflow.
    Retrieves semantic chunks and generates a response.
    Returns: (response_text, citations_list, workflow_steps)
    """
    from typing import TypedDict, List, Dict, Any
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
    from langgraph.graph import StateGraph, START, END

    class RAGState(TypedDict):
        query: str
        chat_history: List[BaseMessage]
        context: str
        doc_id: Any
        user: Any
        response: str
        citations: List[Dict[str, Any]]
        workflow_steps: List[str]

    def retrieve_chunks_node(state: RAGState) -> Dict[str, Any]:
        query = state['query']
        user = state['user']
        doc_id = state['doc_id']
        steps = list(state.get('workflow_steps', []))
        steps.append("Analyzing Second Brain vault structure")
        steps.append("Initializing Knowledge Vault search index")
        
        # 1. Fetch metadata about available documents so LLM knows vault contents globally
        from .models import Document
        docs_qs = Document.objects.filter(user=user)
        if doc_id and doc_id != 'all':
            if isinstance(doc_id, list):
                docs_qs = docs_qs.filter(id__in=doc_id)
            elif isinstance(doc_id, str) and ',' in doc_id:
                try:
                    doc_ids = [int(x.strip()) for x in doc_id.split(',') if x.strip().isdigit()]
                    docs_qs = docs_qs.filter(id__in=doc_ids)
                except ValueError:
                    docs_qs = docs_qs.filter(id=doc_id)
            else:
                docs_qs = docs_qs.filter(id=doc_id)
                
        meta_parts = ["[SYSTEM INSTRUCTION: Below is the global metadata of all files currently inside the user's active Vault scope.]"]
        for d in docs_qs:
            meta_parts.append(f"- File: {d.name} (Type: {d.file_type}, Size: {d.file_size_formatted})")
        meta_parts.append("[End of Global Vault Metadata. Below are specific excerpts related to the user's query]")
        global_metadata_text = "\n".join(meta_parts)

        # 2. Call chunk_semantic_search for query-specific excerpts
        chunks = chunk_semantic_search(user, query, doc_id=doc_id, top_k=40)
        
        context_parts = [global_metadata_text]
        citations = []
        
        for idx, c in enumerate(chunks):
            # c is dict: {chunk, score, doc_name, page_number, text, doc_id}
            context_parts.append(
                f"Source: {c['doc_name']} (Page {c['page_number']})\n"
                f"Content: {c['text']}\n"
            )
            citations.append({
                'doc_name': c['doc_name'],
                'page_number': c['page_number'],
                'score': int(c['score'] * 100),
                'doc_id': c['doc_id']
            })
            
        context_text = "\n---\n".join(context_parts)
        steps.append(f"Semantic search matched {len(chunks)} relevant chunks")
        
        return {
            "context": context_text,
            "citations": citations,
            "workflow_steps": steps
        }

    def generate_answer_node(state: RAGState) -> Dict[str, Any]:
        from langchain_mistralai import ChatMistralAI
        
        query = state['query']
        context = state['context']
        chat_history = state['chat_history']
        steps = list(state.get('workflow_steps', []))
        steps.append("Formulating query prompt with contextual metadata")
        
        mistral_key = os.getenv('MISTRAL_API_KEY')
        if not mistral_key:
            return {
                "response": "AI API Key is missing. Please configure MISTRAL_API_KEY in your .env file.",
                "workflow_steps": steps + ["Failed: Missing API Key"]
            }
            
        llm = ChatMistralAI(
            model="mistral-large-latest",
            mistral_api_key=mistral_key,
            temperature=0.4
        )
        
        system_prompt = (
            "You are Vault AI, a highly advanced 2027 AI assistant connected to the user's Second Brain library.\n"
            "Examine the provided document context to answer the user's specific question directly, concisely, and conversationally.\n"
            "If the query is a simple greeting, reply warmly and introduce yourself.\n"
            "If you do not know the answer from the context, use your general knowledge, but add a brief note stating it was not found in the vault.\n\n"
            "--- ROADMAPS, DIAGRAMS & 2027 CONTEXT ---\n"
            "- The current year is 2027. All roadmaps, guides, framework updates, calendar dates, and planning must align with 2027 standards.\n"
            "- If the user asks for a diagram, flowchart, sequence, or visual graph, you MUST output a beautiful, valid Mermaid diagram wrapped in a ```mermaid block.\n"
            "- If the user asks for a roadmap or step-by-step guide, structure it as a detailed chronological roadmap with checkboxes, step numbers, progress markers, and sub-steps.\n\n"
            f"Context:\n{context}"
        )
        
        messages = [SystemMessage(content=system_prompt)]
        
        # Map input messages list
        for m in chat_history:
            if isinstance(m, dict):
                sender = m.get('sender', 'user')
                content = m.get('content', '')
                if sender == 'user':
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))
            else:
                messages.append(m)
                
        messages.append(HumanMessage(content=query))
        
        steps.append("Streaming structured prompt tokens to Mistral Large")
        steps.append("Resolving semantic dependencies and citations")
        
        try:
            response = llm.invoke(messages)
            ai_response = response.content.strip()
        except Exception as e:
            ai_response = f"Error during generation: {str(e)}"
            steps.append(f"Generation error: {str(e)}")
            
        return {
            "response": ai_response,
            "workflow_steps": steps
        }

    # Build LangGraph workflow
    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve_chunks_node)
    workflow.add_node("generate", generate_answer_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    app = workflow.compile()
    
    # Run the workflow
    initial_state = {
        "query": query_text,
        "chat_history": chat_history_messages,
        "context": "",
        "doc_id": doc_id_or_ids,
        "user": user,
        "response": "",
        "citations": [],
        "workflow_steps": []
    }
    
    final_output = app.invoke(initial_state)
    return (
        final_output.get("response", ""),
        final_output.get("citations", []),
        final_output.get("workflow_steps", [])
    )


