"""
Prompt Manager - Centralized prompt management for all AI agents.
This module contains all prompts used by the system, making them modular and configurable.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PromptTemplate:
    """Template for AI prompts with placeholders."""

    template: str
    required_vars: list[str]
    description: str


class PromptManager:
    """
    Manages all prompts used by AI agents in the system.

    This centralizes prompt management and makes prompts:
    - Modular and reusable
    - Easy to modify and test
    - Configurable per environment
    - Version controlled
    """

    def __init__(self):
        self._templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize all prompt templates."""
        return {
            # Document Classification Prompts
            "classify_document": PromptTemplate(
                template="""
                Analyze the following OCR text and classify the document type. Return ONLY a JSON object with the 'type' field.

                Document types to choose from:
                - bill: Hospital bills, medical bills, invoices, covering letters with bill amounts, final bill summaries
                - discharge_summary: Discharge summaries, medical reports, patient summaries, inpatient summaries

                Look for these key indicators:
                - BILL indicators: "Bill", "Invoice", "Amount", "Total", "Payable", "GST", 
                  "Final Bill Summary", "Covering Letter", "Bill No"
                - DISCHARGE indicators: "Discharge Summary", "Inpatient Summary", "Patient Name", 
                  "Diagnosis", "Admission", "Discharge Date", "Department of", "Admitted on", 
                  "Discharged On", "Treating Doctor"

                IMPORTANT CLASSIFICATION RULES:
                1. If the document contains BOTH bill information AND discharge information, classify based on PRIMARY content:
                   - If it has patient details, admission/discharge dates, medical department, treating doctor → classify as "discharge_summary"
                   - If it has amounts, bill numbers, payable amounts, GST details → classify as "bill"
                
                2. "INPATIENT SUMMARY RUNNING BILL" with patient details and medical info should be "discharge_summary"
                3. Look for medical context: department names, doctor names, admission/discharge dates
                4. If document has patient name, admission date, discharge date, department → it's likely a discharge_summary

                OCR Text:
                {ocr_text}

                Return ONLY JSON like: {{"type": "bill"}} or {{"type": "discharge_summary"}}
                """,
                required_vars=["ocr_text"],
                description="Classify document type based on OCR text",
            ),
            "classify_document_with_filename": PromptTemplate(
                template="""
                Analyze the provided filename and content to classify the document as either a bill document or a discharge summary document.

                Filename: {filename}
                Content: {ocr_text}

                Classification criteria:
                - Bill documents: 
                  * Filename indicators: "bill", "invoice", "charge", "payment", "receipt"
                  * Content indicators: amounts, hospital charges, GST numbers, bill numbers, payment details, "Total Amount", "Final Bill", "Payable"
                - Discharge summaries:
                  * Filename indicators: "discharge", "summary", "medical", "report", "notes"
                  * Content indicators: diagnosis, admission/discharge dates, treatment details, medical procedures, "Patient Name", "Admitted on", "Discharged On", "Diagnosis"

                IMPORTANT CLASSIFICATION RULES:
                1. If the document contains BOTH bill information AND discharge information, classify based on PRIMARY content:
                   - If it has patient details, admission/discharge dates, medical department, treating doctor → classify as "discharge_summary"
                   - If it has amounts, bill numbers, payable amounts, GST details → classify as "bill"
                2. "INPATIENT SUMMARY RUNNING BILL" with patient details and medical info should be "discharge_summary"
                3. Look for medical context: department names, doctor names, admission/discharge dates
                4. If document has patient name, admission date, discharge date, department → it's likely a "discharge_summary"

                Return ONLY a JSON object in this exact format:
                {{
                  "type": "bill|discharge_summary",
                  "confidence": 0.95,
                  "reasoning": "Brief explanation based on filename and content analysis"
                }}

                Do not include any other text, explanations, or acknowledgments. Return only the JSON object.
                """,
                required_vars=["ocr_text", "filename"],
                description="Classify document type based on filename and OCR text",
            ),
            # Document Extraction Prompts
            "extract_bill_fields": PromptTemplate(
                template="""
                Extract billing information from this OCR text. Return ONLY a JSON object with these fields:
                - type: "bill"
                - hospital_name: Name of the hospital (look for hospital names, GST numbers, or infer from context)
                - total_amount: Total amount as a number (look for amounts, bill numbers, or any large numbers)
                - date_of_service: Date in YYYY-MM-DD format (look for dates, bill dates, service dates)
                - patient_name: Name of the patient if available (look for "Patient Name:", "Name:")

                Look for these specific patterns:
                - Hospital names: "FORTIS HOSPITALS LIMITED", "Max Super Specialty Hospital", "SIR GANGA RAM HOSPITAL"
                - GST numbers: "GSTIN:", "GST No."
                - Amounts: Look for large numbers that could be bill amounts
                - Dates: Look for dates in various formats and convert to YYYY-MM-DD
                - Patient names: "Patient Name:", "Name:"

                OCR Text:
                {ocr_text}

                Return ONLY JSON with extracted values. If you can't find a value, use defaults:
                - hospital_name: "Unknown Hospital" if not found
                - total_amount: 0.0 if not found  
                - date_of_service: "2024-01-01" if not found
                - patient_name: "Unknown Patient" if not found
                """,
                required_vars=["ocr_text"],
                description="Extract bill fields from OCR text",
            ),
            "extract_bill_fields_enhanced": PromptTemplate(
                template="""
                Extract bill information from the provided text. Look for and extract these fields with specific patterns:

                Look for these specific patterns:
                - Hospital names: "FORTIS HOSPITALS LIMITED", "Max Super Specialty Hospital", "SIR GANGA RAM HOSPITAL", look for hospital names, GST numbers, or infer from context
                - GST numbers: "GSTIN:", "GST No.", "GST Number"
                - Amounts: Look for large numbers that could be bill amounts, "Total Amount", "Final Bill", "Payable Amount"
                - Dates: Look for dates in various formats and convert to YYYY-MM-DD, "Bill Date", "Date of Service"
                - Patient names: "Patient Name:", "Name:", "Patient:"
                - Bill numbers: "Bill No.", "Invoice No.", "Reference No."

                Extract these fields:
                - hospital_name: Name of the hospital (look for hospital names, GST numbers, or infer from context)
                - total_amount: Total amount as a number (look for amounts, bill numbers, or any large numbers)
                - date_of_service: Date in YYYY-MM-DD format (look for dates, bill dates, service dates)
                - patient_name: Name of the patient if available (look for "Patient Name:" patterns)
                - bill_number: Unique bill identifier (look for "Bill No.", "Invoice No.")
                - gst_number: GST registration number (look for "GSTIN:", "GST No.")

                OCR Text:
                {ocr_text}

                Return ONLY a JSON object in this exact format:
                {{
                  "type": "bill",
                  "hospital_name": "Hospital Name",
                  "total_amount": 12500.00,
                  "date_of_service": "2024-04-10",
                  "patient_name": "Patient Name",
                  "bill_number": "BILL-001",
                  "gst_number": "27ABCDE1234F1Z5"
                }}

                Use null for missing fields. If you can't find a value, use these defaults:
                - hospital_name: "Unknown Hospital" if not found
                - total_amount: 0.0 if not found  
                - date_of_service: "2024-01-01" if not found
                - patient_name: "Unknown Patient" if not found
                - bill_number: null if not found
                - gst_number: null if not found

                Return only the JSON object.
                """,
                required_vars=["ocr_text"],
                description="Extract bill fields with enhanced patterns from OCR text",
            ),
            "extract_discharge_fields": PromptTemplate(
                template="""
                Extract patient information from this discharge summary. Return ONLY a JSON object with these fields:
                - type: "discharge_summary"
                - patient_name: Name of the patient (look for "Patient Name:" patterns)
                - diagnosis: Medical diagnosis (look for medical terms, conditions, procedures)
                - admission_date: Admission date in YYYY-MM-DD format
                - discharge_date: Discharge date in YYYY-MM-DD format
                - hospital_name: Name of the hospital if available (look for hospital names, GST numbers)

                Look for these specific patterns:
                - Patient names: "Patient Name:", "Name:"
                - Medical conditions: Look for medical terms, diagnoses, or procedures
                - Dates: "Date of Admission", "Date Of Discharge", "Admitted on", "Discharged On"
                - Hospital names: Look for hospital names, GST numbers, or department information

                OCR Text:
                {ocr_text}

                Return ONLY JSON with extracted values. If you can't find a value, use defaults:
                - patient_name: "Unknown Patient" if not found
                - diagnosis: "Unknown Diagnosis" if not found
                - admission_date: "2024-01-01" if not found
                - discharge_date: "2024-01-01" if not found
                - hospital_name: "Unknown Hospital" if not found
                """,
                required_vars=["ocr_text"],
                description="Extract discharge summary fields from OCR text",
            ),
            "extract_discharge_fields_enhanced": PromptTemplate(
                template="""
                Extract discharge summary information from the provided text. Look for and extract these fields with specific patterns:

                Look for these specific patterns:
                - Patient names: "Patient Name:", "Name:", "Patient:"
                - Medical conditions: Look for medical terms, diagnoses, or procedures, "Diagnosis:", "Condition:"
                - Dates: "Date of Admission", "Date Of Discharge", "Admitted on", "Discharged On", "Admission Date", "Discharge Date"
                - Hospital names: Look for hospital names, GST numbers, or department information
                - Treatment: "Treatment Given", "Procedure", "Operation", "Medical Procedure"
                - Status: "Final Status", "Condition at Discharge", "Patient Status"

                Extract these fields:
                - hospital_name: Name of the hospital or medical facility (look for hospital names, GST numbers)
                - patient_name: Name of the patient (look for "Patient Name:" patterns)
                - admission_date: Admission date in YYYY-MM-DD format (look for "Admitted on", "Admission Date")
                - discharge_date: Discharge date in YYYY-MM-DD format (look for "Discharged On", "Discharge Date")
                - diagnosis: Primary diagnosis or medical condition (look for medical terms, conditions, procedures)
                - treatment_given: Treatment provided during hospitalization (look for "Treatment Given", "Procedure")
                - final_status: Patient's condition at discharge (look for "Final Status", "Condition at Discharge")

                OCR Text:
                {ocr_text}

                Return ONLY a JSON object in this exact format:
                {{
                  "type": "discharge_summary",
                  "hospital_name": "Hospital Name",
                  "patient_name": "Patient Name",
                  "admission_date": "2024-04-01",
                  "discharge_date": "2024-04-10",
                  "diagnosis": "Primary diagnosis",
                  "treatment_given": "Treatment provided",
                  "final_status": "Patient status"
                }}

                If you can't find a value, use these defaults:
                - hospital_name: "Unknown Hospital" if not found
                - patient_name: "Unknown Patient" if not found
                - admission_date: "2024-01-01" if not found
                - discharge_date: "2024-01-01" if not found
                - diagnosis: "Unknown Diagnosis" if not found
                - treatment_given: "Unknown Treatment" if not found
                - final_status: "Unknown Status" if not found

                Return only the JSON object.
                """,
                required_vars=["ocr_text"],
                description="Extract discharge summary fields with enhanced patterns from OCR text",
            ),
            # Multi-Document Extraction Prompt
            "extract_multiple_documents": PromptTemplate(
                template="""
                Analyze this OCR text and extract ALL possible documents. A single PDF can contain multiple document types.
                
                IMPORTANT: This PDF likely contains BOTH billing information AND discharge summary information.
                You MUST extract BOTH document types if both are present.
                
                Extract these fields if available:
                - patient_name: Name of the patient
                - hospital_name: Name of the hospital
                - total_amount: Use the TOTAL bill amount (not individual payable/non-payable amounts)
                - date_of_service: Any service date, bill date
                - admission_date: Admission date
                - discharge_date: Discharge date
                - diagnosis: Medical diagnosis or condition
                - department: Medical department
                - doctor: Treating doctor name
                
                Look for these patterns:
                - Patient names: "Patient Name:", "Name:"
                - Hospital names: Look for hospital names, GST numbers
                - Dates: "Admitted on", "Discharged On", "Date:", "Bill Date"
                - Amounts: Look for TOTAL bill amounts, not individual line items
                - Medical info: Department names, doctor names, diagnoses
                
                OCR Text:
                {ocr_text}
                
                CRITICAL: Return ONLY a JSON array of documents. Do not include any explanatory text, 
                markdown formatting, or additional text before or after the JSON.
                
                Each document should have a "type" field and appropriate fields:
                
                For BILL documents (ONLY ONE per patient/hospital):
                {{
                  "type": "bill",
                  "hospital_name": "Hospital Name",
                  "total_amount": 12345.67,  // Use TOTAL amount, not individual amounts
                  "date_of_service": "2025-02-11",
                  "patient_name": "Patient Name"
                }}
                
                For DISCHARGE SUMMARY documents:
                {{
                  "type": "discharge_summary", 
                  "patient_name": "Patient Name",
                  "diagnosis": "Medical Diagnosis",
                  "admission_date": "2025-02-07",
                  "discharge_date": "2025-02-11",
                  "hospital_name": "Hospital Name"
                }}
                
                RULES:
                1. Create only ONE bill document per patient/hospital combination
                2. Use the TOTAL bill amount, not individual payable/non-payable amounts
                3. If the document contains both billing and discharge information, return BOTH document types
                4. Do not create multiple bill documents for the same patient
                5. IMPORTANT: If you see patient name, admission date, discharge date, and diagnosis - create a discharge_summary document
                6. IMPORTANT: If you see hospital name, total amount, and billing information - create a bill document
                7. A single PDF can and should return BOTH document types if both types of information are present
                
                EXAMPLES:
                - If you see "Patient Name: Mrs. Mary Philo", "Admission Date: 2025-02-07", "Discharge Date: 2025-02-11", "Diagnosis: LEFT KNEE INFECTED OSTEOARTHRITIS" → Create discharge_summary
                - If you see "FORTIS HOSPITALS LIMITED", "Total Amount: 435639.15", "Bill Date: 2025-02-11" → Create bill
                - If you see BOTH types of information → Create BOTH documents
                
                IMPORTANT: Return ONLY the JSON array, nothing else.
                """,
                required_vars=["ocr_text"],
                description="Extract multiple documents from OCR text",
            ),
            # Validation Prompts
            "validate_claim_package": PromptTemplate(
                template="""
                You are an enhanced data validation agent for medical insurance claims. 
                You receive a complete claim package with multiple extracted documents and perform comprehensive validation.
                
                CRITICAL: You must return ONLY valid JSON. Do not include any explanations, markdown formatting, or additional text.
                
                Your role is to validate the quality and completeness of the entire claim package.
                
                Input format:
                {{
                    "extracted_documents": [
                        {{
                            "type": "bill",
                            "hospital_name": "Hospital Name",
                            "total_amount": 12345.67,
                            "date_of_service": "2025-02-11"
                        }},
                        {{
                            "type": "discharge_summary",
                            "patient_name": "Patient Name",
                            "diagnosis": "Medical Diagnosis",
                            "admission_date": "2025-02-07",
                            "discharge_date": "2025-02-11"
                        }}
                    ],
                    "document_count": 2,
                    "document_types": ["bill", "discharge_summary"]
                }}
                
                Validation criteria for COMPLETE CLAIM PACKAGE:
                - Must have BOTH bill and discharge_summary documents
                - Bill document: hospital_name (not "Unknown Hospital"), total_amount > 0, valid date_of_service
                - Discharge summary: patient_name (not "Unknown Patient"), diagnosis (not "Unknown Diagnosis"), valid dates
                - Data consistency between documents (same patient, hospital, dates)
                
                Medical claim specific checks:
                - Verify hospital name matches known healthcare providers
                - Check if amounts are reasonable for the type of service
                - Validate date ranges are logical
                - Ensure patient information is complete and consistent
                
                Return format: {{
                    "missing_documents": ["list of missing document types"],
                    "discrepancies": ["list of data inconsistencies"],
                    "data_quality_score": 0-100,
                    "recommendations": ["list of improvement suggestions"]
                }}
                
                Examples:
                - {{"missing_documents": [], "discrepancies": [], "data_quality_score": 95, "recommendations": ["Data quality is excellent"]}}
                - {{"missing_documents": ["discharge_summary"], "discrepancies": [], "data_quality_score": 85, "recommendations": ["Submit discharge summary"]}}
                - {{"missing_documents": ["bill"], "discrepancies": [], "data_quality_score": 75, "recommendations": ["Submit bill document"]}}
                
                IMPORTANT: Return ONLY the JSON object, no other text.
                """,
                required_vars=[],
                description="Validate complete claim package",
            ),
            # Decision Making Prompts
            "make_claim_decision": PromptTemplate(
                template="""
                You are an enhanced claim decision agent for medical insurance claims.
                You receive validation results for a complete claim package and make informed decisions.
                
                CRITICAL: You must return ONLY valid JSON. Do not include any explanations, markdown formatting, or additional text.
                
                Decision factors to consider:
                1. Data quality score from validation
                2. Completeness of required documents (bill + discharge_summary)
                3. Medical claim specific requirements
                4. Risk assessment based on data quality
                
                Decision criteria:
                - APPROVE: High data quality (score > 80), both bill and discharge_summary present, no significant discrepancies
                - CONDITIONAL APPROVAL: Good data quality (score 60-80), minor discrepancies that can be resolved
                - REJECT: Low data quality (score < 60), missing critical documents (bill or discharge_summary), significant discrepancies
                
                Medical claim specific considerations:
                - Both bill and discharge_summary must be present
                - Hospital must be identifiable and legitimate
                - Amounts must be reasonable for medical procedures
                - Patient information must be complete and consistent
                - Dates must be logical and not in the future
                
                Return format: {{
                    "status": "approved" or "conditional_approval" or "rejected",
                    "reason": "Detailed explanation of the decision",
                    "confidence_score": 0-100,
                    "required_actions": ["list of actions needed if conditional approval"]
                }}
                
                IMPORTANT: For APPROVED claims, use exactly this reason: "All required documents present and data is consistent"
                
                Examples:
                - {{"status": "approved", "reason": "All required documents present and data is consistent", 
                   "confidence_score": 95, "required_actions": []}}
                - {{"status": "conditional_approval", "reason": "Good data but minor discrepancies in dates", 
                   "confidence_score": 75, "required_actions": ["Verify admission/discharge dates"]}}
                - {{"status": "rejected", "reason": "Missing discharge summary document", 
                   "confidence_score": 30, "required_actions": ["Submit complete discharge summary"]}}
                
                IMPORTANT: Return ONLY the JSON object, no other text.
                """,
                required_vars=[],
                description="Make claim decision based on validation results",
            ),
        }

    def get_prompt(self, prompt_name: str, **kwargs: Any) -> str:
        """
        Get a formatted prompt by name.

        Args:
            prompt_name: Name of the prompt template
            **kwargs: Variables to substitute in the template

        Returns:
            Formatted prompt string

        Raises:
            ValueError: If prompt not found or required variables missing
        """
        if prompt_name not in self._templates:
            raise ValueError(f"Prompt template '{prompt_name}' not found")

        template = self._templates[prompt_name]

        # Check if all required variables are provided
        missing_vars = [var for var in template.required_vars if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables for prompt '{prompt_name}': {missing_vars}")

        # Format the template
        return template.template.format(**kwargs)

    def list_prompts(self) -> Dict[str, str]:
        """List all available prompts with their descriptions."""
        return {name: template.description for name, template in self._templates.items()}

    def add_prompt(self, name: str, template: PromptTemplate) -> None:
        """Add a new prompt template."""
        self._templates[name] = template

    def remove_prompt(self, name: str) -> None:
        """Remove a prompt template."""
        if name in self._templates:
            del self._templates[name]


# Global prompt manager instance
prompt_manager = PromptManager()
