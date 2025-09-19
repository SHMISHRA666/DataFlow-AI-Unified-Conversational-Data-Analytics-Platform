import os
import argparse
from agents.agent_ingestion import DataIngestionAgent
from agents.agent_cleaning import DataCleaningAgent
from agents.agent_transformation import DataTransformationAgent
from agents.agent_analysis import DataAnalysisAgent

def create_directories():
    """Creates the necessary data directories for the workflow."""
    os.makedirs("data/01_raw", exist_ok=True)
    os.makedirs("data/02_cleaned", exist_ok=True)
    os.makedirs("data/03_transformed", exist_ok=True)
    os.makedirs("data/04_analysis_results", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    print("📂 All necessary directories created.")

def get_user_input():
    """Get user input interactively for file upload configuration."""
    print("\n📁 Dataset Upload Configuration")
    print("=" * 40)
    
    # Get file path
    print("\n📝 Choose your dataset file:")
    print("Supported formats: .csv, .json, .xlsx, .xls")
    
    while True:
        file_path = input("\nEnter the path to your dataset file: ").strip()
        if file_path:
            # Check if file exists
            if os.path.exists(file_path):
                # Check file extension
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in ['.csv', '.json', '.xlsx', '.xls']:
                    break
                else:
                    print(f"❌ Unsupported file format: {file_ext}")
                    print("Please use .csv, .json, .xlsx, or .xls files")
            else:
                print(f"❌ File not found: {file_path}")
        else:
            print("❌ Please enter a file path")
    
    # Get file format for processing
    print(f"\n📋 Detected file format: {file_ext}")
    
    # Get number of rows to process (optional)
    while True:
        try:
            limit_input = input("Enter number of rows to process (press Enter for all rows): ").strip()
            if not limit_input:
                limit = None
                break
            limit = int(limit_input)
            if limit > 0:
                break
            print("❌ Please enter a positive number")
        except ValueError:
            print("❌ Please enter a valid number")
    
    return {
        'file_path': file_path,
        'file_format': file_ext,
        'limit': limit
    }

def get_workflow_options():
    """Get user input for workflow configuration."""
    print("\n⚙️  Workflow Configuration")
    print("=" * 30)
    
    # Check if user wants to skip steps
    print("\n🔄 Workflow Steps:")
    print("1. Data Ingestion (load and validate uploaded dataset)")
    print("2. Data Cleaning (remove duplicates, clean text)")
    print("3. Data Transformation (extract years, normalize names)")
    print("4. Data Analysis (generate reports)")
    
    print("\n⏭️  Skip Options:")
    print("You can skip any step if you have existing data files.")
    
    skip_ingestion = input("Skip data ingestion? (y/N): ").strip().lower() == 'y'
    skip_cleaning = input("Skip data cleaning? (y/N): ").strip().lower() == 'y'
    skip_transformation = input("Skip data transformation? (y/N): ").strip().lower() == 'y'
    skip_analysis = input("Skip data analysis? (y/N): ").strip().lower() == 'y'
    
    return {
        'skip_ingestion': skip_ingestion,
        'skip_cleaning': skip_cleaning,
        'skip_transformation': skip_transformation,
        'skip_analysis': skip_analysis
    }

def parse_workflow_args():
    """Parse command line arguments for the workflow."""
    parser = argparse.ArgumentParser(description="Dataset Processing Workflow")
    
    # File Configuration
    parser.add_argument("--file", dest="file_path", default=None,
                       help="Path to dataset file (.csv, .json, .xlsx, .xls)")
    parser.add_argument("--limit", dest="row_limit", type=int, default=None,
                       help="Number of rows to process (default: all rows)")
    
    # Workflow Configuration
    parser.add_argument("--skip-ingestion", action="store_true",
                       help="Skip the ingestion step (use existing raw data)")
    parser.add_argument("--skip-cleaning", action="store_true",
                       help="Skip the cleaning step")
    parser.add_argument("--skip-transformation", action="store_true",
                       help="Skip the transformation step")
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip the analysis step")
    
    # Output Configuration
    parser.add_argument("--output-dir", dest="output_dir", default="data",
                       help="Base directory for output files")
    
    # Interactive mode
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode (guided setup)")
    
    return parser.parse_args()

def run_workflow(args, user_config=None):
    """Run the complete dataset processing workflow."""
    print("\n🚀 Starting Dataset Processing Workflow")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Define file paths
    raw_data_path = f"{args.output_dir}/01_raw/raw_data.csv"
    cleaned_data_path = f"{args.output_dir}/02_cleaned/cleaned_data.csv"
    transformed_data_path = f"{args.output_dir}/03_transformed/transformed_data.csv"
    analysis_output_base = f"{args.output_dir}/04_analysis_results/analysis"
    
    # Initialize agents
    ingestion_agent = DataIngestionAgent()
    cleaning_agent = DataCleaningAgent()
    transformation_agent = DataTransformationAgent()
    analysis_agent = DataAnalysisAgent()
    
    # Step 1: Data Ingestion
    if not args.skip_ingestion:
        print("\n📁 STEP 1: Data Ingestion")
        print("-" * 30)
        
        if user_config:
            # Use file from user input
            file_path = user_config['file_path']
            limit = user_config['limit']
            print(f"📁 Loading dataset from: {file_path}")
        elif args.file_path:
            # Use file from command line
            file_path = args.file_path
            limit = args.row_limit
            print(f"📁 Loading dataset from: {file_path}")
        else:
            print("❌ No dataset file specified. Exiting workflow.")
            return False
        
        # Load the dataset
        df = ingestion_agent.load_dataset(
            file_path=file_path,
            limit=limit,
            output_path=raw_data_path
        )
        
        if df.empty:
            print("❌ No data loaded. Exiting workflow.")
            return False
    else:
        print("⏭️  Skipping data ingestion step")
        if not os.path.exists(raw_data_path):
            print(f"❌ Raw data file not found: {raw_data_path}")
            return False
    
    # Step 2: Data Cleaning
    if not args.skip_cleaning:
        print("\n🧼 STEP 2: Data Cleaning")
        print("-" * 30)
        cleaning_agent.process(input_path=raw_data_path, output_path=cleaned_data_path)
    else:
        print("⏭️  Skipping data cleaning step")
        if not args.skip_transformation and not os.path.exists(cleaned_data_path):
            cleaned_data_path = raw_data_path  # Use raw data for next step
    
    # Step 3: Data Transformation
    if not args.skip_transformation:
        print("\n🔧 STEP 3: Data Transformation")
        print("-" * 30)
        input_path = cleaned_data_path if os.path.exists(cleaned_data_path) else raw_data_path
        transformation_agent.process(input_path=input_path, output_path=transformed_data_path)
    else:
        print("⏭️  Skipping data transformation step")
        if not args.skip_analysis:
            # Determine which file to use for analysis
            if os.path.exists(transformed_data_path):
                pass  # Use transformed data
            elif os.path.exists(cleaned_data_path):
                transformed_data_path = cleaned_data_path
            else:
                transformed_data_path = raw_data_path
    
    # Step 4: Data Analysis
    if not args.skip_analysis:
        print("\n📊 STEP 4: Data Analysis")
        print("-" * 30)
        input_path = transformed_data_path if os.path.exists(transformed_data_path) else raw_data_path
        analysis_agent.analyze(input_path=input_path, output_base_path=analysis_output_base)
    else:
        print("⏭️  Skipping data analysis step")
    
    print("\n🎉 Workflow completed successfully!")
    return True

def main():
    """Main entry point for the workflow."""
    args = parse_workflow_args()
    
    # Check if user wants interactive mode
    if args.interactive or (not args.file_path and not args.skip_ingestion):
        print("🎯 Welcome to the Dataset Processing Workflow!")
        print("This tool will help you process and analyze your custom datasets.")
        
        # Get user configuration
        user_config = get_user_input()
        
        # Get workflow options
        workflow_options = get_workflow_options()
        
        # Update args with user preferences
        args.skip_ingestion = workflow_options['skip_ingestion']
        args.skip_cleaning = workflow_options['skip_cleaning']
        args.skip_transformation = workflow_options['skip_transformation']
        args.skip_analysis = workflow_options['skip_analysis']
        
        # Run workflow with user configuration
        try:
            success = run_workflow(args, user_config)
            return 0 if success else 1
        except KeyboardInterrupt:
            print("\n⚠️  Workflow interrupted by user")
            return 1
        except Exception as e:
            print(f"\n❌ Workflow failed with error: {e}")
            return 1
    
    # Command line mode
    # Validate file path if ingestion is not skipped
    if not args.skip_ingestion and not args.file_path:
        print("❌ Error: Dataset file path is required for data ingestion.")
        print("   Use --file option or run in interactive mode")
        return 1
    
    # Validate file exists if provided
    if args.file_path and not os.path.exists(args.file_path):
        print(f"❌ Error: Dataset file not found: {args.file_path}")
        return 1
    
    try:
        success = run_workflow(args)
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n⚠️  Workflow interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Workflow failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())