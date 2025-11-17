#!/usr/bin/env python3
"""
Text Comparison and Evaluation Tool

This module provides functionality to compare two CSV files containing text data
and calculate various similarity metrics including Levenshtein distance, ROUGE-W scores,
and first mismatch positions.

Usage Examples and Notes:

1. Command Line Usage Examples:
   
   # Basic comparison with default settings
   python diff_optimized.py baseline.csv comparison.csv
   
   # Show only texts with very low ROUGE-W scores
   python diff_optimized.py baseline.csv comparison.csv --rouge-threshold 0.2
   
   # Show texts with early mismatches (within first 5 characters)
   python diff_optimized.py baseline.csv comparison.csv --first-mismatch-threshold 5
   
   # Combine both thresholds (OR logic - either condition triggers inclusion)
   python diff_optimized.py baseline.csv comparison.csv --rouge-threshold 0.4 --first-mismatch-threshold 10
   
   # Specify custom output file
   python diff_optimized.py baseline.csv comparison.csv --output my_results.txt
   
   # Use short form of output parameter
   python diff_optimized.py baseline.csv comparison.csv -o analysis_report.txt

2. CSV File Format:
   
   The CSV files should contain text data in the first column:
   
   baseline.csv:
   "This is the expected output"
   "Another reference text"
   "Final example text"
   
   comparison.csv:
   "This is the actual output"
   "Another comparison text"
   "Final example result"

3. Interpretation of Results:
   
   - ROUGE-W F1 Score: Higher is better (0-1 range), measures word-level similarity
   - Levenshtein Ratio: Higher is better (0-1 range), measures character-level similarity
   - First Mismatch Position: Lower indicates earlier divergence (1-indexed)
   
   Texts are flagged for detailed output if they meet ANY of the threshold conditions:
   - ROUGE-W score below rouge_threshold
   - First mismatch position at or below mismatch_threshold
"""

import argparse
import csv
import importlib
import ssl
import subprocess
import sys
from typing import List, Optional, Dict, Any


def check_and_install_dependencies() -> None:
    """
    Check and install required Python packages and NLTK data.
    
    This function ensures all necessary dependencies are available:
    - py-rouge: For ROUGE score calculation
    - nltk: For natural language processing
    - python-Levenshtein: For edit distance calculation
    """
    required_packages = ['py-rouge', 'nltk', 'python-Levenshtein']
    
    for package in required_packages:
        try:
            # Convert package names to module names for import testing
            module_name = package.replace('py-', '').replace('python-', '')
            importlib.import_module(module_name)
            print(f"✓ {package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    # Download required NLTK data
    _download_nltk_data()


def _download_nltk_data() -> None:
    """
    Download required NLTK data files.
    
    Downloads the punkt_tab tokenizer data required for text processing.
    Handles SSL certificate issues that may occur during download.
    """
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt_tab')
            print("✓ NLTK punkt_tab data is already available")
        except LookupError:
            # Handle SSL certificate issues
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            print("Downloading NLTK punkt_tab data...")
            nltk.download('punkt_tab')
    except ImportError:
        # NLTK should be installed by now, import it again
        import nltk
        print("Downloading NLTK punkt_tab data...")
        nltk.download('punkt_tab')


def read_csv_file(file_path: str) -> List[str]:
    """
    Read text data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of text strings from the first column of the CSV file
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        csv.Error: If there's an error reading the CSV file
    """
    text_results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if row:  # Skip empty rows
                    text_results.append(row[0])
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    return text_results


class TextComparator:
    """
    A class for comparing texts using various similarity metrics.
    
    This class provides comprehensive text comparison functionality with configurable
    thresholds for different metrics. It can identify texts that differ significantly
    based on ROUGE-W scores and/or early character mismatches.
    """
    
    def __init__(self, rouge_threshold: float = 0.5, mismatch_threshold: Optional[int] = None):
        """
        Initialize the TextComparator.
        
        Args:
            rouge_threshold: Threshold for ROUGE-W score below which
                           detailed comparison results are printed (default: 0.5)
            mismatch_threshold: Threshold for first mismatch position below which
                              detailed comparison results are printed. If None,
                              mismatch position filtering is disabled (default: None)
                              
        Examples:
            # Only show texts with ROUGE-W score below 0.3
            comparator = TextComparator(rouge_threshold=0.3)
            
            # Only show texts with first mismatch within first 10 characters
            comparator = TextComparator(mismatch_threshold=10)
            
            # Show texts that meet either condition
            comparator = TextComparator(rouge_threshold=0.4, mismatch_threshold=15)
        """
        self.rouge_threshold = rouge_threshold
        self.mismatch_threshold = mismatch_threshold
        self._setup_evaluator()
    
    def _setup_evaluator(self) -> None:
        """Setup the ROUGE evaluator."""
        # Import here to ensure dependencies are installed
        import rouge
        self.evaluator = rouge.Rouge(metrics=['rouge-w'], limit_length=False)
        
    def _calculate_first_mismatch_position(self, text1: str, text2: str) -> Optional[int]:
        """
        Calculate the position of the first character mismatch between two texts.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Position (1-indexed) of first mismatch, or None if texts are identical
        """
        len1, len2 = len(text1), len(text2)
        min_len = min(len1, len2)
        
        # Find first differing character
        for i in range(min_len):
            if text1[i] != text2[i]:
                return i + 1
        
        # If one string is longer than the other
        if len1 != len2:
            return min_len + 1
        
        return None

    
    def compare_texts(self, base_texts: List[str], 
                     comparison_texts: List[str]) -> Dict[str, Any]:
        """
        Compare two lists of texts and calculate similarity metrics.
        
        Args:
            base_texts: List of reference texts
            comparison_texts: List of texts to compare against reference
            
        Returns:
            Dictionary containing comparison results and statistics
            
        Raises:
            ValueError: If the input lists have different lengths
        """
        if len(base_texts) != len(comparison_texts):
            raise ValueError(
                f"Input lists must have the same length. "
                f"Got {len(base_texts)} base texts and {len(comparison_texts)} comparison texts."
            )
        
        # Import Levenshtein here to ensure it's installed
        import Levenshtein
        
        lev_distances = []
        first_mismatch_positions = []
        detailed_comparisons = []
        
        for i, (base_text, comp_text) in enumerate(zip(base_texts, comparison_texts)):
            # Calculate normalized Levenshtein distance (similarity ratio)
            lev_ratio = Levenshtein.ratio(base_text, comp_text)
            lev_distances.append(lev_ratio)
            
            # Calculate first mismatch position
            first_mismatch_pos = self._calculate_first_mismatch_position(base_text, comp_text)
            if first_mismatch_pos is not None:
                first_mismatch_positions.append(first_mismatch_pos)
            
            # Calculate ROUGE-W score
            rouge_score = self.evaluator.get_scores([base_text], [comp_text])['rouge-w']['f']
            
            # Determine if this comparison should be included in detailed results
            include_in_details = False
            triggered_by_rouge = False
            triggered_by_mismatch = False
            
            # Check ROUGE-W threshold
            if rouge_score <= self.rouge_threshold:
                include_in_details = True
                triggered_by_rouge = True
            
            # Check first mismatch threshold (if enabled)
            if (self.mismatch_threshold is not None and first_mismatch_pos is not None
                    and first_mismatch_pos <= self.mismatch_threshold):
                include_in_details = True
                triggered_by_mismatch = True
            
            # Store detailed comparison if any threshold condition is met
            if include_in_details:
                detailed_comparisons.append({
                    'index': i,
                    'base_text': base_text,
                    'comparison_text': comp_text,
                    'levenshtein_ratio': lev_ratio,
                    'first_mismatch_position': first_mismatch_pos,
                    'rouge_w_f1': rouge_score,
                    'triggered_by_rouge': triggered_by_rouge,
                    'triggered_by_mismatch': triggered_by_mismatch
                })
        
        # Calculate overall ROUGE scores
        overall_rouge_scores = self.evaluator.get_scores(comparison_texts, base_texts)
        
        # Calculate averages
        avg_first_mismatch_position = (
            sum(first_mismatch_positions) / len(first_mismatch_positions)
            if first_mismatch_positions else -1
        )
        avg_lev_distance = 1 - (sum(lev_distances) / len(lev_distances))
        avg_rouge_w_f1 = overall_rouge_scores['rouge-w']['f']
        
        return {
            'total_comparisons': len(base_texts),
            'detailed_comparisons': detailed_comparisons,
            'statistics': {
                'avg_first_mismatch_position': avg_first_mismatch_position,
                'avg_levenshtein_distance': avg_lev_distance,
                'avg_rouge_w_f1': avg_rouge_w_f1,
                'overall_rouge_scores': overall_rouge_scores['rouge-w']
            }
        }
    
    def write_results(self, results: Dict[str, Any], output_file: Optional[str] = None) -> None:
        """
        Print comparison results in a formatted way.
        
        Args:
            results: Results dictionary from compare_texts method
            output_file: Optional file path to write results. If None, writes to default file.
        """
        # Determine output file path
        if output_file is None:
            output_file = "comparison_results.txt"
        
        # Prepare output content
        output_lines = []
        # Collect detailed comparisons for texts below threshold
        for comp in results['detailed_comparisons']:
            output_lines.append(f"Base Text: {comp['base_text']}\n")
            output_lines.append(f"Comparison Text: {comp['comparison_text']}\n")
            
            # Show which conditions triggered the inclusion
            triggers = []
            if comp.get('triggered_by_rouge', False):
                triggers.append(f"ROUGE-W < {self.rouge_threshold}")
            if comp.get('triggered_by_mismatch', False):
                triggers.append(f"First mismatch ≤ {self.mismatch_threshold}")
            
            trigger_info = f" (Triggered by: {', '.join(triggers)})" if triggers else ""
            
            output_lines.append(f"Levenshtein ratio: {comp['levenshtein_ratio']:.4f}, "
                              f"First mismatch position: {comp['first_mismatch_position']}, "
                              f"Rouge-W F1: {comp['rouge_w_f1']:.4f}{trigger_info}")
            output_lines.append("----------------------------------\n\n")
        
        # Collect summary statistics
        stats = results['statistics']
        output_lines.append(f"Total comparisons: {results['total_comparisons']}")
        output_lines.append(f"Comparisons below threshold: {len(results['detailed_comparisons'])}")
        output_lines.append(f"Overall ROUGE-W scores: {stats['overall_rouge_scores']}")
        output_lines.append("\n***********************************")
        output_lines.append(f"avg_first_mismatch_position\t{stats['avg_first_mismatch_position']:.4f}")
        output_lines.append(f"avg_levenshtein_distance\t{stats['avg_levenshtein_distance']:.4f}")
        output_lines.append(f"avg_rouge_w_f1\t{stats['avg_rouge_w_f1']:.4f}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in output_lines:
                f.write(line + '\n')
        print(f"Results written to: {output_file}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Compare two CSV files containing text data using various similarity metrics."
    )
    parser.add_argument(
        'csv_file1',
        help='Path to the first CSV file (reference/base texts)'
    )
    parser.add_argument(
        'csv_file2', 
        help='Path to the second CSV file (comparison texts)'
    )
    parser.add_argument(
        '--rouge-threshold',
        type=float,
        default=0.5,
        help='ROUGE-W threshold below which detailed results are printed (default: 0.5)'
    )
    parser.add_argument(
        '--first-mismatch-threshold',
        type=int,
        default=None,
        help='First mismatch position threshold below which detailed results are printed. '
             'If not specified, mismatch position filtering is disabled.'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default="comparison_results.txt",
        help='Output file path for results. If not specified, results will be written to '
             'a default file named "comparison_results.txt".'
    )
    
    return parser.parse_args()


def check_diff(base_texts, comparison_texts, rouge_threshold, mismatch_threshold, output_file) -> None:
    """
    Main function to execute the text comparison workflow.
    """
    # Check and install dependencies
    print("Checking dependencies...")
    check_and_install_dependencies()
    
    # Initialize comparator with custom thresholds
    comparator = TextComparator(
        rouge_threshold=rouge_threshold,
        mismatch_threshold=mismatch_threshold
    )
    
    # Print configuration info
    print(f"Configuration:")
    print(f"  ROUGE-W threshold: {comparator.rouge_threshold}")
    print(f"  First mismatch threshold: {comparator.mismatch_threshold}")
    
    # Perform comparison
    print("Performing text comparison...")
    results = comparator.compare_texts(base_texts, comparison_texts)
    
    # # Determine output file path
    # output_file = args.output if args.output else "comparison_results.txt"
    
    # Write results to file
    comparator.write_results(results, output_file)


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    base_texts = read_csv_file(args.csv_file1)
    comparison_texts = read_csv_file(args.csv_file2)
    check_diff(base_texts, comparison_texts, args.rouge_threshold, args.first_mismatch_threshold, args.output)
    