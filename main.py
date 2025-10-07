# main.py – 100% NetworkX Graph-First (FIXED MultiMCP)

from utils.utils import log_step, log_error
import asyncio
from dotenv import load_dotenv
from agentLoop.flow import AgentLoop4
from pathlib import Path
import sys
import os

# Ensure UTF-8 console on Windows to avoid 'charmap' codec errors
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
if os.name == "nt":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass

BANNER = """
──────────────────────────────────────────────────────
🔸  Agentic Query Assistant  🔸
Files first, then your question.
Type 'exit' or 'quit' to leave.
──────────────────────────────────────────────────────
"""

def get_file_input():
    """Get file paths from user"""
    log_step("📁 File Input (optional):", symbol="")
    print("Enter file paths (one per line), or press Enter to skip:")
    print("Example: /path/to/file.csv")
    print("Press Enter twice when done.")
    
    uploaded_files = []
    file_manifest = []
    
    while True:
        file_path = input("📄 File path: ").strip()
        if not file_path:
            break
        
        # Strip quotes from drag-and-drop paths
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]
        
        if Path(file_path).exists():
            uploaded_files.append(file_path)
            file_manifest.append({
                "path": file_path,
                "name": Path(file_path).name,
                "size": Path(file_path).stat().st_size
            })
            print(f"✅ Added: {Path(file_path).name}")
        else:
            print(f"❌ File not found: {file_path}")
    
    return uploaded_files, file_manifest

def get_user_query():
    """Get query from user"""
    log_step("📝 Your Question:", symbol="")
    return input().strip()

async def main():
    load_dotenv()
    print(BANNER)
    
    # Initialize AgentLoop4 without MCP layer
    log_step("🚀 Initializing DataFlow AI")
    agent_loop = AgentLoop4(None)
    
    while True:
        try:
            # Get file input first
            uploaded_files, file_manifest = get_file_input()
            
            # Get user query
            query = get_user_query()
            if query.lower() in ['exit', 'quit']:
                break
            
            # Process with AgentLoop4 - returns ExecutionContextManager object
            log_step("🔄 Processing with AgentLoop4...")
            execution_context = await agent_loop.run(query, file_manifest, uploaded_files)
            
            # Minimal completion message
            print("\n" + "="*60)
            print("✅ DataFlow AI completed.")
            print("="*60)
            
            print("\n😴 Agent Resting now")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            log_error(f"Error: {e}")
            print("Let's try again...")
        
        # Continue prompt
        cont = input("\nContinue? (press Enter) or type 'exit': ").strip()
        if cont.lower() in ['exit', 'quit']:
            break

    # No MCP shutdown required

if __name__ == "__main__":
    asyncio.run(main())
