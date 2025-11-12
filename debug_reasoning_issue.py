#!/usr/bin/env python3
"""
Debug script to check what TinfoilAgent is actually generating for reasoning sections
"""

import sys
import json
sys.path.append('/Users/choemanseung/789/hft')

try:
    from mlx_models.tinfoilAgent import TinfoilAgent
    TINFOIL_AVAILABLE = True
except ImportError:
    print("❌ TinfoilAgent not available")
    TINFOIL_AVAILABLE = False
    sys.exit(1)

from generate_triage_dialogues import (
    build_triage_lookup,
    create_grounded_triage_prompt,
    call_tinfoil_agent
)


def debug_single_generation():
    """Generate one dialogue and show the raw response."""
    
    # Build triage lookup
    triage_lookup = build_triage_lookup()
    
    if not triage_lookup:
        print("❌ No triage data found")
        return
    
    # Pick the first triage entry
    first_entry = list(triage_lookup.values())[0]
    print(f"🔬 Testing with condition: {first_entry['condition']}")
    print(f"   Triage level: {first_entry['triage_level']}")
    
    # Generate prompt for case 6 (new variation)
    prompt = create_grounded_triage_prompt(first_entry, 6)
    
    print(f"\n📝 Generated prompt:")
    print("-" * 80)
    print(prompt)
    print("-" * 80)
    
    # Call TinfoilAgent
    print(f"\n🤖 Calling TinfoilAgent...")
    response = call_tinfoil_agent(prompt)
    
    if response:
        print(f"\n📋 Raw TinfoilAgent response:")
        print("=" * 80)
        print(response)
        print("=" * 80)
        
        # Check for reasoning sections manually
        reasoning_patterns = [
            "Reasoning for Clarifying Question (CoT):",
            "Reasoning for Clarifying Question:",
            "Reasoning for Triage Decision (CoT):",
            "Reasoning for Triage Decision:",
            "CoT):",
            "reasoning",
            "Clarifying Question Reasoning",
            "Triage Decision Reasoning"
        ]
        
        print(f"\n🔍 Looking for reasoning patterns:")
        for pattern in reasoning_patterns:
            if pattern.lower() in response.lower():
                print(f"   ✅ Found: '{pattern}'")
            else:
                print(f"   ❌ Missing: '{pattern}'")
        
        # Show sections split by lines
        print(f"\n📑 Response sections:")
        lines = response.split('\n')
        for i, line in enumerate(lines):
            if line.strip():
                print(f"   {i+1:2d}: {line}")
    
    else:
        print("❌ No response from TinfoilAgent")


if __name__ == "__main__":
    debug_single_generation()