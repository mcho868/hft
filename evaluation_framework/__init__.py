"""
Medical Triage System - Comprehensive Evaluation Framework

This package provides a complete evaluation system for medical triage models,
testing all combinations of RAG configurations with trained adapters.

Components:
- config_matrix_generator: Generates RAG × adapter configuration matrix
- evaluation_pipeline: Core evaluation engine with resume capabilities  
- metrics_collector: Medical-specific accuracy, safety, and performance metrics
- appropriateness_judge: LLM-as-judge clinical appropriateness evaluation
- performance_profiler: Memory usage and inference speed profiling
- results_analyzer: Statistical analysis and visualization dashboard
- master_evaluation_runner: Orchestrates complete evaluation pipeline

Quick Start:
    from evaluation_framework.master_evaluation_runner import MasterEvaluationRunner
    
    runner = MasterEvaluationRunner("/path/to/base/directory")
    results = runner.run_complete_evaluation()

Command Line:
    python master_evaluation_runner.py --max-combinations 100
"""

__version__ = "1.0.0"
__author__ = "Medical Triage Evaluation Team"

# Import main components for easy access
from .master_evaluation_runner import MasterEvaluationRunner
from .config_matrix_generator import ConfigMatrixGenerator
from .evaluation_pipeline import EvaluationPipeline
from .metrics_collector import MedicalMetricsCollector
from .appropriateness_judge import ClinicalAppropriatenessJudge
from .performance_profiler import PerformanceProfiler
from .results_analyzer import ResultsAnalyzer

__all__ = [
    "MasterEvaluationRunner",
    "ConfigMatrixGenerator", 
    "EvaluationPipeline",
    "MedicalMetricsCollector",
    "ClinicalAppropriatenessJudge",
    "PerformanceProfiler",
    "ResultsAnalyzer"
]