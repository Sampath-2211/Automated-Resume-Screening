"""
Resume PDF to JSON Converter
Converts PDF resumes to structured JSON for fine-tuning
"""
import json
import os
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz  # PyMuPDF

# Optional: Use LLM for better extraction
USE_LLM_EXTRACTION = os.environ.get('USE_LLM_EXTRACTION', 'false').lower() == 'true'


@dataclass
class ResumeData:
    """Structured resume data"""
    id: str
    filename: str
    raw_text: str
    clean_text: str
    sections: Dict[str, str]
    extracted_skills: List[str]
    extracted_education: List[Dict[str, str]]
    extracted_experience: List[Dict[str, str]]
    contact_info: Dict[str, str]
    word_count: int
    extraction_method: str
    

class ResumeExtractor:
    """Extract structured data from resume PDFs"""
    
    # Common section headers in resumes
    SECTION_PATTERNS = {
        'summary': r'(?:summary|objective|profile|about\s*me)',
        'experience': r'(?:experience|employment|work\s*history|professional\s*experience)',
        'education': r'(?:education|academic|qualification|degree)',
        'skills': r'(?:skills|technical\s*skills|competencies|expertise|technologies)',
        'projects': r'(?:projects|personal\s*projects|academic\s*projects)',
        'certifications': r'(?:certification|certificate|licenses|accreditation)',
        'achievements': r'(?:achievement|accomplishment|award|honor)',
    }
    
    # Common skills to look for
    TECH_SKILLS = [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'rust', 'php', 'swift', 'kotlin',
        'react', 'angular', 'vue', 'node', 'express', 'django', 'flask', 'fastapi', 'spring', 'rails',
        'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'dynamodb',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'git', 'ci/cd',
        'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
        'html', 'css', 'sass', 'tailwind', 'bootstrap', 'graphql', 'rest api', 'microservices',
        'agile', 'scrum', 'jira', 'confluence', 'linux', 'unix', 'bash', 'powershell',
    ]
    
    EMAIL_PATTERN = r'[\w\.-]+@[\w\.-]+\.\w+'
    PHONE_PATTERN = r'[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}'
    LINKEDIN_PATTERN = r'linkedin\.com/in/[\w-]+'
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract raw text from PDF"""
        text_parts = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text_parts.append(page.get_text("text"))
        return "\n".join(text_parts)
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        return text.strip()
    
    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information"""
        contact = {}
        
        email_match = re.search(self.EMAIL_PATTERN, text, re.IGNORECASE)
        if email_match:
            contact['email'] = email_match.group()
            
        phone_match = re.search(self.PHONE_PATTERN, text)
        if phone_match:
            contact['phone'] = phone_match.group()
            
        linkedin_match = re.search(self.LINKEDIN_PATTERN, text, re.IGNORECASE)
        if linkedin_match:
            contact['linkedin'] = linkedin_match.group()
            
        return contact
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract resume sections using regex patterns"""
        sections = {}
        text_lower = text.lower()
        
        # Find all section positions
        section_positions = []
        for section_name, pattern in self.SECTION_PATTERNS.items():
            matches = list(re.finditer(pattern, text_lower))
            for match in matches:
                section_positions.append((match.start(), section_name, match.group()))
        
        # Sort by position
        section_positions.sort(key=lambda x: x[0])
        
        # Extract content between sections
        for i, (pos, name, _) in enumerate(section_positions):
            start = pos
            end = section_positions[i + 1][0] if i + 1 < len(section_positions) else len(text)
            content = text[start:end].strip()
            
            # Remove the header itself from content
            lines = content.split('\n')
            if len(lines) > 1:
                content = '\n'.join(lines[1:]).strip()
            
            if name not in sections or len(content) > len(sections.get(name, '')):
                sections[name] = content[:2000]  # Limit section size
                
        return sections
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text"""
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.TECH_SKILLS:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(skill)
                
        return found_skills
    
    def extract_experience_entries(self, text: str) -> List[Dict[str, str]]:
        """Extract work experience entries"""
        experiences = []
        
        # Look for patterns like "Company Name | Role | Date"
        # Or "Role at Company (Date - Date)"
        date_pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|\d{4})\s*[-–to]+\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|\d{4}|present|current)'
        
        matches = re.finditer(date_pattern, text, re.IGNORECASE)
        for match in matches:
            # Get context around the date (likely contains company/role)
            start = max(0, match.start() - 200)
            end = min(len(text), match.end() + 500)
            context = text[start:end]
            
            experiences.append({
                'date_range': match.group(),
                'context': context.strip()[:500]
            })
            
        return experiences[:10]  # Limit to 10 entries
    
    def extract_education_entries(self, text: str) -> List[Dict[str, str]]:
        """Extract education entries"""
        education = []
        
        # Common degree patterns
        degree_patterns = [
            r"(bachelor'?s?|master'?s?|ph\.?d\.?|b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?|b\.?e\.?|m\.?e\.?|b\.?tech|m\.?tech)",
            r"(computer science|engineering|mathematics|physics|business|economics)",
        ]
        
        for pattern in degree_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 200)
                context = text[start:end]
                
                education.append({
                    'match': match.group(),
                    'context': context.strip()[:300]
                })
                
        return education[:5]  # Limit entries
    
    def extract_with_llm(self, text: str) -> Dict[str, Any]:
        """Use LLM for better extraction (optional)"""
        if not self.llm:
            return {}
            
        prompt = f"""Extract structured information from this resume.

RESUME TEXT:
{text[:4000]}

Return ONLY valid JSON (no comments):
{{
  "name": "candidate name if found",
  "skills": ["skill1", "skill2"],
  "experience_years": estimated total years,
  "current_role": "most recent job title",
  "education_level": "highest degree",
  "key_achievements": ["achievement1", "achievement2"]
}}

JSON:"""
        
        try:
            return self.llm.generate_json(prompt, max_tokens=512)
        except:
            return {}
    
    def process_single_resume(self, pdf_path: str) -> ResumeData:
        """Process a single resume PDF"""
        path = Path(pdf_path)
        
        # Generate unique ID
        resume_id = hashlib.md5(path.name.encode()).hexdigest()[:12]
        
        # Extract text
        raw_text = self.extract_text_from_pdf(pdf_path)
        clean_text = self.clean_text(raw_text)
        
        # Extract structured data
        sections = self.extract_sections(raw_text)
        skills = self.extract_skills(raw_text)
        experience = self.extract_experience_entries(raw_text)
        education = self.extract_education_entries(raw_text)
        contact = self.extract_contact_info(raw_text)
        
        # Optional LLM extraction
        extraction_method = "regex"
        if USE_LLM_EXTRACTION and self.llm:
            llm_data = self.extract_with_llm(clean_text)
            if llm_data:
                skills = list(set(skills + llm_data.get('skills', [])))
                extraction_method = "regex+llm"
        
        return ResumeData(
            id=resume_id,
            filename=path.name,
            raw_text=raw_text,
            clean_text=clean_text,
            sections=sections,
            extracted_skills=skills,
            extracted_education=education,
            extracted_experience=experience,
            contact_info=contact,
            word_count=len(clean_text.split()),
            extraction_method=extraction_method
        )
    
    def process_directory(
        self, 
        input_dir: str, 
        output_file: str,
        max_workers: int = 4,
        progress_callback=None
    ) -> Dict[str, Any]:
        """Process all PDFs in a directory"""
        
        input_path = Path(input_dir)
        pdf_files = list(input_path.glob("**/*.pdf"))
        
        print(f"Found {len(pdf_files)} PDF files")
        
        results = []
        errors = []
        
        for i, pdf_file in enumerate(pdf_files):
            try:
                resume_data = self.process_single_resume(str(pdf_file))
                results.append(asdict(resume_data))
                
                if progress_callback:
                    progress_callback(i + 1, len(pdf_files), pdf_file.name)
                elif (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(pdf_files)}")
                    
            except Exception as e:
                errors.append({'file': str(pdf_file), 'error': str(e)})
                
        # Save results
        output = {
            'total_processed': len(results),
            'total_errors': len(errors),
            'resumes': results,
            'errors': errors
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
        print(f"\nCompleted: {len(results)} resumes saved to {output_file}")
        print(f"Errors: {len(errors)}")
        
        return output


def create_training_pairs(
    resumes_json: str,
    job_descriptions: List[Dict[str, str]],
    output_file: str
) -> None:
    """
    Create training pairs for sentence transformer fine-tuning
    
    Args:
        resumes_json: Path to converted resumes JSON
        job_descriptions: List of {"title": "...", "description": "..."}
        output_file: Output path for training pairs
    """
    with open(resumes_json, 'r') as f:
        data = json.load(f)
    
    resumes = data['resumes']
    pairs = []
    
    for jd in job_descriptions:
        jd_text = jd['description']
        jd_title = jd['title'].lower()
        
        for resume in resumes:
            resume_text = resume['clean_text'][:1500]
            skills = resume['extracted_skills']
            
            # Simple relevance scoring based on skill overlap and keywords
            # You should manually label some of these for better quality
            
            jd_skills = set(word.lower() for word in jd_text.split() if len(word) > 3)
            resume_skills = set(s.lower() for s in skills)
            
            overlap = len(jd_skills & resume_skills)
            
            # Heuristic score (replace with manual labels for better results)
            if overlap >= 5:
                score = 0.8 + (min(overlap, 10) - 5) * 0.04  # 0.8 - 1.0
            elif overlap >= 2:
                score = 0.4 + (overlap - 2) * 0.13  # 0.4 - 0.8
            else:
                score = overlap * 0.2  # 0.0 - 0.4
            
            pairs.append({
                'resume_id': resume['id'],
                'resume_text': resume_text,
                'jd_title': jd['title'],
                'jd_text': jd_text[:1500],
                'relevance_score': round(score, 2),
                'skill_overlap': overlap,
                'is_labeled': False  # Mark as auto-generated
            })
    
    # Save pairs
    with open(output_file, 'w') as f:
        json.dump({'pairs': pairs, 'total': len(pairs)}, f, indent=2)
    
    print(f"Created {len(pairs)} training pairs")
    print(f"Saved to {output_file}")


# CLI usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python resume_to_json_converter.py <input_dir> <output_file>")
        print("Example: python resume_to_json_converter.py ./resumes ./resumes_converted.json")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    
    extractor = ResumeExtractor()
    extractor.process_directory(input_dir, output_file)