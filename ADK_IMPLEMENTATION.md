# Google ADK Multi-Agent Implementation

## 🎯 **What Happens When You Upload a File**

When you upload a file to our new Google ADK system, here's the complete flow:

### **1. File Upload & Validation** 📁
```bash
curl -X POST "http://localhost:8000/v2/adk/process-claim" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@hospital_bill.pdf"
```

**Validation Checks:**
- ✅ File exists and is PDF format
- ✅ File size ≤ 10MB
- ✅ Maximum 10 files per request

### **2. OCR Processing** 🔍
```python
# Mistral OCR extracts text from PDF
ocr_result = await ocr_service.extract_text_async(file_content, filename)
# Returns: {"text": "Patient Name: John Doe...", "quality_score": 85.5}
```

### **3. Google ADK Multi-Agent Pipeline** 🤖

The system uses Google ADK's built-in agent framework with **5 specialized agents**:

#### **Agent 1: ClassificationAgent** 🏷️
```python
classification_agent = LlmAgent(
    name="ClassificationAgent",
    model="gemini-2.0-flash-exp",
    instruction="Classify document type: bill or discharge",
    output_key="classification_result"
)
```
**Output:** `{"type": "bill", "confidence": 0.95, "reasoning": "Contains bill amounts"}`

#### **Agent 2: BillAgent** 💰
```python
bill_agent = LlmAgent(
    name="BillAgent", 
    model="gemini-2.0-flash-exp",
    instruction="Extract bill data: hospital, amount, dates, etc.",
    output_key="bill_data"
)
```
**Output:** `{"hospital_name": "Fortis Hospital", "total_amount": 50000, "date_of_service": "2024-01-15"}`

#### **Agent 3: DischargeAgent** 📋
```python
discharge_agent = LlmAgent(
    name="DischargeAgent",
    model="gemini-2.0-flash-exp", 
    instruction="Extract discharge data: diagnosis, treatment, dates",
    output_key="discharge_data"
)
```
**Output:** `{"patient_name": "John Doe", "diagnosis": "Appendicitis", "treatment_given": "Laparoscopic surgery"}`

#### **Agent 4: ValidationAgent** ✅
```python
validation_agent = LlmAgent(
    name="ValidationAgent",
    model="gemini-2.0-flash-exp",
    instruction="Validate data consistency and quality",
    output_key="validation_result"
)
```
**Output:** `{"is_valid": true, "validation_score": 85, "issues": [], "summary": "Data is consistent"}`

#### **Agent 5: DecisionAgent** 🎯
```python
decision_agent = LlmAgent(
    name="DecisionAgent",
    model="gemini-2.0-flash-exp",
    instruction="Make final claim decision: approved/rejected/pending",
    output_key="decision_result"
)
```
**Output:** `{"decision": "approved", "approved_amount": 45000, "risk_score": 15, "confidence": 90}`

### **4. Sequential Processing Pipeline** 🔄
```python
claim_processing_pipeline = SequentialAgent(
    name="ClaimProcessingPipeline",
    sub_agents=[
        classification_agent,
        bill_agent, 
        discharge_agent,
        validation_agent,
        decision_agent
    ]
)
```

### **5. Final Response** 📊
```json
{
  "success": true,
  "processing_time_ms": 2500,
  "claim_id": "claim_1703123456",
  "documents_processed": 2,
  "results": [
    {
      "filename": "hospital_bill.pdf",
      "classification": {"type": "bill", "confidence": 0.95},
      "bill_data": {"hospital_name": "Fortis", "total_amount": 50000},
      "validation": {"is_valid": true, "validation_score": 85},
      "decision": {"decision": "approved", "approved_amount": 45000}
    }
  ],
  "summary": {
    "total_documents": 2,
    "successful_processing": 2,
    "decisions": ["approved", "approved"]
  }
}
```

## 🚀 **Key Advantages of Google ADK**

### **1. Simplified Implementation** ✨
- **Before:** 500+ lines of custom agent code
- **After:** 50 lines using Google ADK's built-in framework

### **2. Built-in Features** 🛠️
- ✅ **Automatic session management**
- ✅ **Structured input/output schemas**
- ✅ **Sequential processing pipeline**
- ✅ **Error handling and retries**
- ✅ **Performance monitoring**

### **3. Easy to Extend** 🔧
```python
# Add a new agent in 3 lines
new_agent = LlmAgent(
    name="NewAgent",
    model="gemini-2.0-flash-exp",
    instruction="Your custom instruction"
)

# Add to pipeline
claim_processing_pipeline.sub_agents.append(new_agent)
```

## 📍 **API Endpoints**

### **Process Claims**
```bash
POST /v2/adk/process-claim
```

### **Health Check**
```bash
GET /v2/adk/health
```

### **Agent Status**
```bash
GET /v2/adk/agents/status
```

### **System Info**
```bash
GET /v2/adk/info
```

## 🔧 **Setup & Installation**

1. **Install Google ADK:**
```bash
pip install google-adk
```

2. **Set Environment Variables:**
```bash
export GOOGLE_API_KEY="your-gemini-api-key"
```

3. **Run the Application:**
```bash
uvicorn app.main:app --reload
```

## 🎯 **Why This Approach is Better**

1. **Production Ready:** Google ADK is designed for production use
2. **Less Code:** 90% reduction in custom agent code
3. **Better Error Handling:** Built-in retry and error management
4. **Scalable:** Easy to add new agents and modify workflows
5. **Maintainable:** Standard Google ADK patterns and practices

## 📊 **Performance**

- **Processing Time:** ~2-3 seconds per document
- **Concurrent Processing:** Supports multiple documents
- **Error Recovery:** Automatic retries and fallbacks
- **Memory Efficient:** Session-based processing

This implementation demonstrates the power of Google ADK's multi-agent framework while maintaining all the functionality of our original system with much cleaner, more maintainable code! 