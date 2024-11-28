import argparse
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def main():
    parser = argparse.ArgumentParser(
        description="Performance Evaluation for Fine-tuned CLIP Model",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluation.py --performance both
  python scripts/evaluation.py --performance task
  python scripts/evaluation.py --performance human
        """
    )
    parser.add_argument(
        '--performance',
        type=str,
        choices=['task', 'human', 'both'],
        default='both',
        help="Choose which performance to evaluate:\n  'task'  - Task-specific performance\n  'human' - Correlation with human scores\n  'both'  - Evaluate both 'task' and 'human' performances (default)"
    )
    
    args = parser.parse_args()
    
    performance = args.performance
    
    if performance in ['human', 'both']:
        print("Starting Human Correlation Performance Evaluation...")
        from utils.human_correlation_performance import run_human_correlation_evaluation
        run_human_correlation_evaluation()
        print("Human Correlation Performance Evaluation Finished.\n")
    
    if performance in ['task', 'both']:
        print("Starting Task Performance Evaluation...")
        from utils.task_performance import main
        main()
        print("Task Performance Evaluation Finished.\n")
    
    print("All selected performance evaluations are completed.")

if __name__ == "__main__":
    main()
