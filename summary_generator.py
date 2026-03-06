"""
Automated Resume Screening - Summary Generator
Adaptive summaries with inline citations based on evaluation results.
Used as fallback/helper alongside Node 4 LLM-based summary generation.

Author: Sampath Krishna Tekumalla
"""
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import re

# =============================================================================
# ENUMS AND CONFIG
# =============================================================================
class ToneLevel(Enum):
    EXCELLENT = "excellent"
    STRONG = "strong"
    AVERAGE = "average"
    BELOW = "below"

@dataclass
class SummaryConfig:
    min_sentences: int = 2
    max_sentences: int = 6
    include_citations: bool = True
    include_score: bool = True
    include_recommendation: bool = True

# =============================================================================
# TONE DETERMINER
# =============================================================================
class ToneDeterminer:
    @staticmethod
    def determine(score: int, critical_met: int, critical_total: int) -> ToneLevel:
        if score >= 80 and critical_met == critical_total:
            return ToneLevel.EXCELLENT
        elif score >= 60 and critical_met >= critical_total - 1:
            return ToneLevel.STRONG
        elif score >= 40 or critical_met >= critical_total - 1:
            return ToneLevel.AVERAGE
        else:
            return ToneLevel.BELOW

    @staticmethod
    def get_tone_descriptor(tone: ToneLevel) -> Dict[str, str]:
        descriptors = {
            ToneLevel.EXCELLENT: {
                "opener": "Excellent candidate",
                "connector": "demonstrates exceptional",
                "strength_verb": "excels in",
                "gap_intro": "Minor area for growth:",
                "recommendation": "Strongly recommended for immediate interview"
            },
            ToneLevel.STRONG: {
                "opener": "Strong candidate",
                "connector": "shows solid",
                "strength_verb": "demonstrates",
                "gap_intro": "Development opportunity:",
                "recommendation": "Recommended for interview consideration"
            },
            ToneLevel.AVERAGE: {
                "opener": "Candidate shows potential",
                "connector": "has foundational",
                "strength_verb": "shows capability in",
                "gap_intro": "Key gap identified:",
                "recommendation": "May be considered with development plan"
            },
            ToneLevel.BELOW: {
                "opener": "Candidate does not meet requirements",
                "connector": "lacks sufficient",
                "strength_verb": "has limited",
                "gap_intro": "Critical gap:",
                "recommendation": "Not recommended at this time"
            }
        }
        return descriptors[tone]

# =============================================================================
# LENGTH DETERMINER
# =============================================================================
class LengthDeterminer:
    @staticmethod
    def determine_sentences(tone: ToneLevel, strengths_count: int, gaps_count: int, score: int) -> int:
        if tone == ToneLevel.EXCELLENT and gaps_count == 0:
            return 2
        if tone == ToneLevel.BELOW and strengths_count == 0:
            return 2
        if 38 <= score <= 42 or 58 <= score <= 62:
            return 5
        if strengths_count > 0 and gaps_count > 0:
            return 4
        return 3

# =============================================================================
# SUMMARY BUILDER
# =============================================================================
class SummaryBuilder:
    def __init__(self, config: SummaryConfig = None):
        self.config = config or SummaryConfig()

    def build(self, candidate_name: str, job_title: str, score: int,
              tone: ToneLevel, strengths: List[Dict], gaps: List[Dict],
              num_sentences: int) -> str:
        descriptors = ToneDeterminer.get_tone_descriptor(tone)
        sentences = []

        opener = f"{descriptors['opener']} for the {job_title} position"
        if self.config.include_score:
            opener += f", achieving a score of {score}/100"
        opener += "."
        sentences.append(opener)

        if strengths and len(sentences) < num_sentences:
            for s in strengths[:2]:
                if len(sentences) >= num_sentences:
                    break
                criterion = s.get('criterion', '')
                citations = []
                for cr in s.get('citation_results', []):
                    if cr.get('valid') and cr.get('citation'):
                        citations.append(cr['citation'])
                if citations and self.config.include_citations:
                    cite_text = citations[0][:80]
                    sentence = f"{candidate_name} {descriptors['strength_verb']} {criterion.lower()}, evidenced by <cite>{cite_text}</cite>."
                else:
                    sentence = f"{candidate_name} {descriptors['strength_verb']} {criterion.lower()}."
                sentences.append(sentence)

        if gaps and tone in [ToneLevel.AVERAGE, ToneLevel.BELOW] and len(sentences) < num_sentences:
            gap = gaps[0]
            sentences.append(f"{descriptors['gap_intro']} {gap.get('criterion', '').lower()} requires further development.")

        if self.config.include_recommendation and len(sentences) < num_sentences:
            sentences.append(descriptors['recommendation'] + ".")

        return " ".join(sentences)

    def format_citations_for_display(self, summary: str) -> str:
        def replace_cite(match):
            text = match.group(1)
            escaped = text.replace('"', '&quot;').replace("'", "&#39;")
            return (
                f'<span class="citation-link" data-citation="{escaped}" '
                f'style="background:#fef3c7;padding:2px 4px;border-radius:3px;'
                f'cursor:pointer;border-bottom:2px solid #f59e0b;color:#1e293b !important;">{text}</span>'
            )
        return re.sub(r'<cite>(.*?)</cite>', replace_cite, summary)

# =============================================================================
# ADAPTIVE SUMMARY GENERATOR
# =============================================================================
class AdaptiveSummaryGenerator:
    def __init__(self, config: SummaryConfig = None):
        self.config = config or SummaryConfig()
        self.builder = SummaryBuilder(self.config)

    def generate(self, candidate_name: str, job_title: str,
                 scores_detail: List[Dict], final_score: int) -> Dict[str, Any]:
        strengths = [s for s in scores_detail if s.get('validated_score', 0) >= 4]
        gaps = [s for s in scores_detail if s.get('validated_score', 0) <= 2 and s.get('critical', False)]

        critical_total = sum(1 for s in scores_detail if s.get('critical', False))
        critical_met = sum(1 for s in scores_detail
                          if s.get('critical', False) and s.get('validated_score', 0) >= 3)

        tone = ToneDeterminer.determine(final_score, critical_met, critical_total)
        num_sentences = LengthDeterminer.determine_sentences(tone, len(strengths), len(gaps), final_score)

        summary = self.builder.build(
            candidate_name=candidate_name, job_title=job_title, score=final_score,
            tone=tone, strengths=strengths, gaps=gaps, num_sentences=num_sentences
        )

        display_summary = self.builder.format_citations_for_display(summary)

        return {
            "raw_summary": summary,
            "display_summary": display_summary,
            "tone": tone.value,
            "sentence_count": num_sentences,
            "strengths_count": len(strengths),
            "gaps_count": len(gaps),
            "critical_met": critical_met,
            "critical_total": critical_total,
            "recommendation": ToneDeterminer.get_tone_descriptor(tone)["recommendation"]
        }

    def generate_comparison_summary(self, candidate_name: str, naive_score: int,
                                     validated_score: int, invalidated_citations: List[str]) -> str:
        diff = naive_score - validated_score
        if diff == 0:
            return (
                f"{candidate_name}'s score remained at {validated_score}/100 after "
                f"citation validation. All claimed evidence was verified in the resume."
            )
        summary = (
            f"{candidate_name}'s score was adjusted from {naive_score} to "
            f"{validated_score} (-{diff} points) after citation validation. "
        )
        if invalidated_citations:
            summary += "The following claims could not be verified: "
            summary += "; ".join([f'"{c[:50]}..."' for c in invalidated_citations[:2]])
            summary += "."
        return summary

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================
def generate_summary_for_candidate(candidate_data: Dict, job_title: str,
                                    config: SummaryConfig = None) -> Dict[str, Any]:
    generator = AdaptiveSummaryGenerator(config)
    return generator.generate(
        candidate_name=candidate_data.get('name', 'Candidate'),
        job_title=job_title,
        scores_detail=candidate_data.get('scores_detail', []),
        final_score=candidate_data.get('score', 0)
    )

def get_recommendation_from_score(score: int, critical_met: int, critical_total: int) -> str:
    tone = ToneDeterminer.determine(score, critical_met, critical_total)
    return ToneDeterminer.get_tone_descriptor(tone)["recommendation"]

__all__ = [
    'ToneLevel', 'SummaryConfig', 'ToneDeterminer', 'LengthDeterminer',
    'SummaryBuilder', 'AdaptiveSummaryGenerator',
    'generate_summary_for_candidate', 'get_recommendation_from_score'
]