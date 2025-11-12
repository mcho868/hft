#!/usr/bin/env python3
"""
Overnight Comparison Runner

This script runs both standard and safety-enhanced triage fine-tuning scripts
sequentially for comprehensive comparison. Designed for overnight execution
with robust progress saving and recovery capabilities.
"""

import subprocess
import json
import os
import time
import csv
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
import traceback

class OvernightRunner:
    def __init__(self, resume_session=None):
        self.resume_mode = resume_session is not None
        
        if self.resume_mode:
            # Resume existing session
            self.session_id = resume_session
            self.results_dir = f"overnight_comparison_{self.session_id}"
            self.progress_file = f"{self.results_dir}/progress.json"
            self.log_file = f"{self.results_dir}/execution_log.txt"
            self.summary_file = f"{self.results_dir}/comparison_summary.json"
            
            # Load existing progress
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
            
            # Correct progress based on actual completed adapters
            actual_standard, actual_safety = count_completed_adapters()
            self.progress["standard_script"]["experiments_completed"] = max(
                self.progress["standard_script"].get("experiments_completed", 0),
                actual_standard
            )
            self.progress["safety_script"]["experiments_completed"] = max(
                self.progress["safety_script"].get("experiments_completed", 0),
                actual_safety
            )
            self.progress["standard_script"]["total_experiments"] = 36
            self.progress["safety_script"]["total_experiments"] = 24
            
            self.start_time = datetime.fromisoformat(self.progress["start_time"])
            print(f"🔄 Resuming session: {self.session_id}")
            print(f"📁 Found {actual_standard}/36 standard adapters, {actual_safety}/24 safety adapters")
        else:
            # New session
            self.start_time = datetime.now()
            self.session_id = self.start_time.strftime("%Y%m%d_%H%M%S")
            self.results_dir = f"overnight_comparison_{self.session_id}"
            self.progress_file = f"{self.results_dir}/progress.json"
            self.log_file = f"{self.results_dir}/execution_log.txt"
            self.summary_file = f"{self.results_dir}/comparison_summary.json"
            
            # Create results directory
            os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize progress tracking (only for new sessions)
        if not self.resume_mode:
            self.progress = {
                "session_id": self.session_id,
                "start_time": self.start_time.isoformat(),
                "status": "starting",
                "current_phase": None,
                "standard_script": {
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "results_dir": None,
                    "experiments_completed": 0,
                    "total_experiments": 36,
                    "successful_experiments": 0,
                    "failed_experiments": 0
                },
                "safety_script": {
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "results_dir": None,
                    "experiments_completed": 0,
                    "total_experiments": 24,
                    "successful_experiments": 0,
                    "safety_passed_experiments": 0
                },
                "interruptions": [],
                "estimated_completion": None,
                "total_runtime": None
            }
        
        self.save_progress()
        self.setup_signal_handlers()
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful interruption"""
        signal.signal(signal.SIGINT, self.handle_interruption)
        signal.signal(signal.SIGTERM, self.handle_interruption)
        
    def handle_interruption(self, signum, frame):
        """Handle interruption gracefully"""
        self.log("⚠️ Interruption signal received. Saving progress...")
        self.progress["interruptions"].append({
            "time": datetime.now().isoformat(),
            "signal": signum,
            "phase": self.progress["current_phase"]
        })
        self.progress["status"] = "interrupted"
        self.save_progress()
        self.log("💾 Progress saved. Safe to exit.")
        sys.exit(1)
    
    def log(self, message):
        """Log message to both console and file"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
            f.flush()
    
    def save_progress(self):
        """Save current progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def estimate_completion_time(self):
        """Estimate completion time based on current progress"""
        if self.progress["current_phase"] == "standard":
            # Estimate based on standard script progress
            completed = self.progress["standard_script"]["experiments_completed"]
            total = self.progress["standard_script"]["total_experiments"]
            if completed > 0:
                elapsed = datetime.now() - datetime.fromisoformat(self.progress["standard_script"]["start_time"])
                avg_time_per_exp = elapsed / completed
                remaining_exp = total - completed + self.progress["safety_script"]["total_experiments"]
                estimated_remaining = avg_time_per_exp * remaining_exp
                return datetime.now() + estimated_remaining
        elif self.progress["current_phase"] == "safety":
            # Similar estimation for safety script
            completed = self.progress["safety_script"]["experiments_completed"]
            total = self.progress["safety_script"]["total_experiments"]
            if completed > 0:
                elapsed = datetime.now() - datetime.fromisoformat(self.progress["safety_script"]["start_time"])
                avg_time_per_exp = elapsed / completed
                remaining_exp = total - completed
                estimated_remaining = avg_time_per_exp * remaining_exp
                return datetime.now() + estimated_remaining
        
        return None
    
    def run_standard_script(self):
        """Run the standard triage fine-tuning script"""
        self.log("🚀 Starting Standard Triage Fine-tuning")
        self.log("=" * 60)
        
        self.progress["current_phase"] = "standard"
        self.progress["standard_script"]["status"] = "running"
        self.progress["standard_script"]["start_time"] = datetime.now().isoformat()
        
        # Count total experiments (6 models × 6 configs = 36)
        self.progress["standard_script"]["total_experiments"] = 36
        self.save_progress()
        
        try:
            # Run standard script
            cmd = ["python", "triage_lora_finetune.py"]
            self.log(f"🏃 Executing: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Monitor output and track progress
            experiment_count = 0
            success_count = 0
            fail_count = 0
            
            for line in process.stdout:
                self.log(line.rstrip())
                
                # Track experiment progress
                if "Starting triage training:" in line:
                    experiment_count += 1
                    self.progress["standard_script"]["experiments_completed"] = experiment_count
                    self.save_progress()
                    
                if "Training SUCCESS" in line:
                    success_count += 1
                    self.progress["standard_script"]["successful_experiments"] = success_count
                    
                if "Training FAILED" in line:
                    fail_count += 1
                    self.progress["standard_script"]["failed_experiments"] = fail_count
                
                # Update estimated completion
                estimated = self.estimate_completion_time()
                if estimated:
                    self.progress["estimated_completion"] = estimated.isoformat()
                    
                self.save_progress()
            
            process.wait()
            return_code = process.returncode
            
            self.progress["standard_script"]["end_time"] = datetime.now().isoformat()
            
            if return_code == 0:
                self.progress["standard_script"]["status"] = "completed"
                self.log("✅ Standard script completed successfully")
                
                # Find and record results directory
                for item in os.listdir('.'):
                    if item.startswith('triage_experiment_results_'):
                        self.progress["standard_script"]["results_dir"] = item
                        break
            else:
                self.progress["standard_script"]["status"] = "failed"
                self.log(f"❌ Standard script failed with return code {return_code}")
                
        except Exception as e:
            self.progress["standard_script"]["status"] = "error"
            self.log(f"❌ Standard script error: {e}")
            self.log(traceback.format_exc())
        
        self.save_progress()
    
    def run_safety_script(self):
        """Run the safety-enhanced triage fine-tuning script"""
        self.log("\n🛡️ Starting Safety-Enhanced Triage Fine-tuning")
        self.log("=" * 60)
        
        self.progress["current_phase"] = "safety"
        self.progress["safety_script"]["status"] = "running"
        self.progress["safety_script"]["start_time"] = datetime.now().isoformat()
        
        # Count total experiments (6 models × 4 configs = 24)
        self.progress["safety_script"]["total_experiments"] = 24
        self.save_progress()
        
        try:
            # Run safety script
            cmd = ["python", "safety_enhanced_triage_finetune.py"]
            self.log(f"🏃 Executing: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Monitor output and track progress
            experiment_count = 0
            safety_passed_count = 0
            
            for line in process.stdout:
                self.log(line.rstrip())
                
                # Track experiment progress
                if "Starting SAFETY-ENHANCED triage training:" in line:
                    experiment_count += 1
                    self.progress["safety_script"]["experiments_completed"] = experiment_count
                    self.save_progress()
                    
                if "SAFETY CONSTRAINTS PASSED" in line:
                    safety_passed_count += 1
                    self.progress["safety_script"]["safety_passed_experiments"] = safety_passed_count
                
                # Update estimated completion
                estimated = self.estimate_completion_time()
                if estimated:
                    self.progress["estimated_completion"] = estimated.isoformat()
                    
                self.save_progress()
            
            process.wait()
            return_code = process.returncode
            
            self.progress["safety_script"]["end_time"] = datetime.now().isoformat()
            
            if return_code == 0:
                self.progress["safety_script"]["status"] = "completed"
                self.log("✅ Safety script completed successfully")
                
                # Find and record results directory
                for item in os.listdir('.'):
                    if item.startswith('safety_triage_results_'):
                        self.progress["safety_script"]["results_dir"] = item
                        break
            else:
                self.progress["safety_script"]["status"] = "failed"
                self.log(f"❌ Safety script failed with return code {return_code}")
                
        except Exception as e:
            self.progress["safety_script"]["status"] = "error"
            self.log(f"❌ Safety script error: {e}")
            self.log(traceback.format_exc())
        
        self.save_progress()
    
    def generate_comparison_summary(self):
        """Generate comprehensive comparison summary"""
        self.log("\n📊 Generating Comparison Summary")
        
        summary = {
            "session_info": {
                "session_id": self.session_id,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_runtime_hours": (datetime.now() - self.start_time).total_seconds() / 3600
            },
            "standard_script_results": self.progress["standard_script"].copy(),
            "safety_script_results": self.progress["safety_script"].copy(),
            "comparison": {
                "total_experiments": (
                    self.progress["standard_script"]["total_experiments"] + 
                    self.progress["safety_script"]["total_experiments"]
                ),
                "completed_experiments": (
                    self.progress["standard_script"]["experiments_completed"] + 
                    self.progress["safety_script"]["experiments_completed"]
                ),
                "standard_success_rate": 0,
                "safety_pass_rate": 0
            }
        }
        
        # Calculate success rates
        if self.progress["standard_script"]["experiments_completed"] > 0:
            summary["comparison"]["standard_success_rate"] = (
                self.progress["standard_script"]["successful_experiments"] / 
                self.progress["standard_script"]["experiments_completed"]
            )
        
        if self.progress["safety_script"]["experiments_completed"] > 0:
            summary["comparison"]["safety_pass_rate"] = (
                self.progress["safety_script"]["safety_passed_experiments"] / 
                self.progress["safety_script"]["experiments_completed"]
            )
        
        # Load detailed results if available
        try:
            if self.progress["standard_script"]["results_dir"]:
                std_progress_file = f"{self.progress['standard_script']['results_dir']}/progress.json"
                if os.path.exists(std_progress_file):
                    with open(std_progress_file, 'r') as f:
                        summary["standard_detailed_results"] = json.load(f)
        except Exception as e:
            self.log(f"⚠️ Could not load standard detailed results: {e}")
        
        try:
            if self.progress["safety_script"]["results_dir"]:
                safety_progress_file = f"{self.progress['safety_script']['results_dir']}/safety_progress.json"
                if os.path.exists(safety_progress_file):
                    with open(safety_progress_file, 'r') as f:
                        summary["safety_detailed_results"] = json.load(f)
        except Exception as e:
            self.log(f"⚠️ Could not load safety detailed results: {e}")
        
        # Save summary
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def print_final_report(self, summary):
        """Print final comparison report"""
        self.log("\n" + "🏆" * 25 + " OVERNIGHT COMPARISON COMPLETE " + "🏆" * 25)
        
        total_time = datetime.now() - self.start_time
        hours = total_time.total_seconds() / 3600
        
        self.log(f"⏱️  Total Runtime: {hours:.2f} hours")
        self.log(f"📊 Session ID: {self.session_id}")
        
        self.log("\n📈 STANDARD SCRIPT RESULTS:")
        std = self.progress["standard_script"]
        self.log(f"   Status: {std['status']}")
        self.log(f"   Experiments: {std['experiments_completed']}/{std['total_experiments']}")
        self.log(f"   Successful: {std['successful_experiments']}")
        self.log(f"   Failed: {std['failed_experiments']}")
        if std["results_dir"]:
            self.log(f"   Results: {std['results_dir']}")
        
        self.log("\n🛡️ SAFETY SCRIPT RESULTS:")
        safety = self.progress["safety_script"]
        self.log(f"   Status: {safety['status']}")
        self.log(f"   Experiments: {safety['experiments_completed']}/{safety['total_experiments']}")
        self.log(f"   Safety Passed: {safety['safety_passed_experiments']}")
        if safety["results_dir"]:
            self.log(f"   Results: {safety['results_dir']}")
        
        self.log("\n⚖️ COMPARISON:")
        comp = summary["comparison"]
        self.log(f"   Total Experiments: {comp['completed_experiments']}/{comp['total_experiments']}")
        self.log(f"   Standard Success Rate: {comp['standard_success_rate']:.2%}")
        self.log(f"   Safety Pass Rate: {comp['safety_pass_rate']:.2%}")
        
        self.log(f"\n💾 Results Summary: {self.summary_file}")
        self.log(f"📋 Full Log: {self.log_file}")
        self.log(f"🔄 Progress File: {self.progress_file}")
        
        if self.progress["interruptions"]:
            self.log(f"\n⚠️ Interruptions: {len(self.progress['interruptions'])}")
    
    def should_skip_phase(self, phase):
        """Check if a phase should be skipped based on completion status"""
        if phase == "standard":
            completed = self.progress["standard_script"]["experiments_completed"]
            total = self.progress["standard_script"]["total_experiments"] 
            return completed >= total
        elif phase == "safety":
            completed = self.progress["safety_script"]["experiments_completed"] 
            total = self.progress["safety_script"]["total_experiments"]
            return completed >= total
        return False
    
    def run(self):
        """Main execution method with resume capability"""
        try:
            if self.resume_mode:
                self.log("🔄 Resuming Overnight Comparison Runner")
                self.log(f"📊 Previous progress:")
                self.log(f"   Standard: {self.progress['standard_script']['experiments_completed']}/{self.progress['standard_script']['total_experiments']} completed")
                self.log(f"   Safety: {self.progress['safety_script']['experiments_completed']}/{self.progress['safety_script']['total_experiments']} completed")
            else:
                self.log("🌙 Starting Overnight Comparison Runner")
                self.log(f"🕐 Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            self.log(f"📁 Results Directory: {self.results_dir}")
            self.log(f"💾 Progress will be saved to: {self.progress_file}")
            
            self.progress["status"] = "running"
            self.save_progress()
            
            # Run standard script (unless completed)
            if not self.should_skip_phase("standard"):
                self.run_standard_script()
                
                # Small break between scripts
                self.log("\n⏸️ Taking 30-second break between scripts...")
                time.sleep(30)
            else:
                self.log("✅ Standard script already completed, skipping...")
            
            # Run safety script (unless completed)
            if not self.should_skip_phase("safety"):
                self.run_safety_script()
            else:
                self.log("✅ Safety script already completed, skipping...")
            
            # Generate final summary
            self.progress["status"] = "completed"
            self.progress["total_runtime"] = (datetime.now() - self.start_time).total_seconds()
            self.save_progress()
            
            summary = self.generate_comparison_summary()
            self.print_final_report(summary)
            
        except Exception as e:
            self.log(f"❌ Critical error in main execution: {e}")
            self.log(traceback.format_exc())
            self.progress["status"] = "error"
            self.save_progress()

def count_completed_adapters():
    """Count actually completed adapters from directories"""
    standard_adapters = 0
    safety_adapters = 0
    
    # Count standard adapters
    if os.path.exists("triage_adapters"):
        for item in os.listdir("triage_adapters"):
            if item.startswith("adapter_triage_") and os.path.isdir(f"triage_adapters/{item}"):
                # Check if adapter has final safetensors file (completed training)
                if os.path.exists(f"triage_adapters/{item}/adapters.safetensors"):
                    standard_adapters += 1
    
    # Count safety adapters
    if os.path.exists("safety_triage_adapters"):
        for item in os.listdir("safety_triage_adapters"):
            if item.startswith("adapter_safe_triage_") and os.path.isdir(f"safety_triage_adapters/{item}"):
                # Check if adapter has final safetensors file (completed training)
                if os.path.exists(f"safety_triage_adapters/{item}/adapters.safetensors"):
                    safety_adapters += 1
    
    return standard_adapters, safety_adapters

def find_incomplete_sessions():
    """Find incomplete overnight sessions that can be resumed"""
    incomplete_sessions = []
    
    for item in os.listdir('.'):
        if item.startswith('overnight_comparison_') and os.path.isdir(item):
            progress_file = f"{item}/progress.json"
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        progress = json.load(f)
                    
                    # Check if session is incomplete
                    if progress.get("status") not in ["completed"]:
                        session_id = progress.get("session_id")
                        if session_id:
                            # Get actual adapter counts
                            actual_standard, actual_safety = count_completed_adapters()
                            
                            # Update progress with actual counts if they're higher
                            progress["standard_script"]["experiments_completed"] = max(
                                progress["standard_script"].get("experiments_completed", 0),
                                actual_standard
                            )
                            progress["safety_script"]["experiments_completed"] = max(
                                progress["safety_script"].get("experiments_completed", 0), 
                                actual_safety
                            )
                            progress["standard_script"]["total_experiments"] = 36
                            progress["safety_script"]["total_experiments"] = 24
                            
                            incomplete_sessions.append({
                                "session_id": session_id,
                                "dir": item,
                                "progress": progress,
                                "actual_standard_completed": actual_standard,
                                "actual_safety_completed": actual_safety
                            })
                except Exception:
                    continue
    
    return incomplete_sessions

def main():
    """Main function with resume capability"""
    print("🌙 Overnight Triage Fine-tuning Comparison Runner")
    print("=" * 60)
    print("This script will run both standard and safety-enhanced fine-tuning")
    print("scripts sequentially. Designed for overnight execution.")
    print()
    print("Features:")
    print("• Robust progress saving and recovery")
    print("• Real-time logging to file and console")
    print("• Graceful interruption handling")
    print("• Comprehensive comparison summary")
    print("• Estimated completion time tracking")
    print("• Resume capability for interrupted sessions")
    print()
    
    # Check for incomplete sessions
    incomplete_sessions = find_incomplete_sessions()
    
    if incomplete_sessions:
        print("🔍 Found incomplete sessions:")
        for i, session in enumerate(incomplete_sessions):
            progress = session["progress"]
            std_progress = progress.get("standard_script", {})
            safety_progress = progress.get("safety_script", {})
            
            print(f"  {i+1}. Session: {session['session_id']}")
            print(f"     Standard: {std_progress.get('experiments_completed', 0)}/36 completed (📁 {session['actual_standard_completed']} adapters found)")
            print(f"     Safety: {safety_progress.get('experiments_completed', 0)}/24 completed (📁 {session['actual_safety_completed']} adapters found)")
            print(f"     Status: {progress.get('status', 'unknown')}")
        print()
        
        resume_choice = input("🔄 Resume incomplete session? (y/N): ").lower()
        if resume_choice in ['y', 'yes']:
            if len(incomplete_sessions) == 1:
                chosen_session = incomplete_sessions[0]
            else:
                try:
                    choice_idx = int(input(f"Which session to resume? (1-{len(incomplete_sessions)}): ")) - 1
                    chosen_session = incomplete_sessions[choice_idx]
                except (ValueError, IndexError):
                    print("❌ Invalid choice")
                    return
            
            print(f"🔄 Resuming session: {chosen_session['session_id']}")
            runner = OvernightRunner(resume_session=chosen_session["session_id"])
            runner.run()
            return
    
    # Check if scripts exist
    required_scripts = ["triage_lora_finetune.py", "safety_enhanced_triage_finetune.py"]
    missing_scripts = [script for script in required_scripts if not os.path.exists(script)]
    
    if missing_scripts:
        print(f"❌ Missing required scripts: {missing_scripts}")
        print("Please ensure both fine-tuning scripts are in the current directory.")
        return
    
    print("✅ All required scripts found")
    print()
    
    # Estimated runtime warning
    print("⏰ ESTIMATED RUNTIME: 6-12 hours")
    print("   Standard Script: ~3-6 hours (36 experiments)")
    print("   Safety Script: ~3-6 hours (24 experiments)")
    print()
    
    response = input("🚀 Ready to start NEW overnight comparison? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Execution cancelled")
        return
    
    # Run new comparison
    runner = OvernightRunner()
    runner.run()

if __name__ == "__main__":
    main()