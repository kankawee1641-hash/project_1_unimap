#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deslopify CLI & Analyzer (Zero-dependency Python tool)
Scans text, markdown, or code for AI writing tropes, clichés, and statistical fingerprints.
"""

import sys
import os
import re
import argparse

# Force UTF-8 for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Common high-confidence AI slop patterns
SLOP_KEYWORDS = {
    # High-severity AI buzzwords
    "delve": "Academic pretension / LLM crutch",
    "delving": "Academic pretension / LLM crutch",
    "testament": "Overdramatic cliché ('serves as a testament to...')",
    "tapestry": "Grandiose purple prose ('rich tapestry of...')",
    "beacon": "Grandiose metaphor ('beacon of innovation')",
    "seamless": "Empty corporate buzzword",
    "seamlessly": "Empty corporate buzzword",
    "powerhouse": "Overhyped superlative",
    "revolutionize": "Unearned marketing hype",
    "revolutionizing": "Unearned marketing hype",
    "game-changer": "Overused marketing idiom",
    "linchpin": "Stilted corporate jargon",
    "cornerstone": "Stilted corporate jargon",
    "unlock": "Infomercial verb ('unlock your potential')",
    "embark": "Purple prose ('embark on an exciting journey')",
    "foster": "Stilted passive verb ('foster collaboration')",
    "harness": "Stilted verb ('harness the power of')",
    "pivotal": "Overused escalation adjective",
    "paramount": "Overused escalation adjective",
    "groundbreaking": "Unearned marketing hype",
    "cutting-edge": "Tired tech cliché"
}

SLOP_PHRASES = [
    (r"\bnot\s+only\b.+?\bbut\s+(?:also\s+)?", "Negative parallelism ('not only X, but Y')"),
    (r"\bit\s+is\s+worth\s+noting\s+that\b", "Filler padding ('it is worth noting that')"),
    (r"\bit\s+is\s+important\s+to\s+note\s+that\b", "Filler padding ('it is important to note that')"),
    (r"\bat\s+its\s+core\b", "Philosophical pontification ('at its core')"),
    (r"\bin\s+today's\s+(?:fast-paced\s+)?(?:digital\s+)?landscape\b", "Tech cliché ('in today's landscape')"),
    (r"\bwhether\s+you're\s+a\s+.+?\s+or\s+a\b", "False inclusivity pattern ('whether you're a X or Y')"),
    (r"\blook\s+no\s+further\b", "Infomercial cliché ('look no further')"),
    (r"\blet's\s+dive\s+in\b", "Conversational cheerleading ('let's dive in')"),
    (r"\bhere's\s+the\s+kicker\b", "Artificial conversational filler ('here's the kicker')"),
    (r"\bserves?\s+as\s+a\s+testament\b", "Overused dramatic trope ('serves as a testament')"),
    (r"\ba\s+rich\s+tapestry\b", "Purple prose ('a rich tapestry')"),
    (r"\bfast,\s+scalable,?\s+and\s+secure\b", "Predictable triadic marketing cliché"),
]

def analyze_text(text):
    """Scan text and return detailed slop findings."""
    lines = text.split("\n")
    findings = []
    
    in_code_block = False
    
    for line_idx, line in enumerate(lines, start=1):
        # Ignore code blocks
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
            
        line_clean = line.strip()
        if not line_clean:
            continue
            
        # 1. Check Phrase Patterns
        for pattern, desc in SLOP_PHRASES:
            matches = re.finditer(pattern, line_clean, re.IGNORECASE)
            for m in matches:
                findings.append({
                    "line": line_idx,
                    "term": m.group(0),
                    "description": desc,
                    "context": line_clean[:100]
                })
                
        # 2. Check Single Keywords
        words = re.findall(r"\b[A-Za-z-]+\b", line_clean)
        for w in words:
            w_lower = w.lower()
            if w_lower in SLOP_KEYWORDS:
                findings.append({
                    "line": line_idx,
                    "term": w,
                    "description": SLOP_KEYWORDS[w_lower],
                    "context": line_clean[:100]
                })
                
        # 3. Check Excessive Em-dash Density (more than 1 per line)
        em_count = line.count("—") + line.count(" -- ")
        if em_count >= 2:
            findings.append({
                "line": line_idx,
                "term": f"{em_count} em-dashes",
                "description": "Excessive em-dash density (rhythm crutch)",
                "context": line_clean[:100]
            })

    total_words = len(re.findall(r"\w+", text))
    if total_words == 0:
        slop_score = 0.0
    else:
        # Score calculation: weighted count per 1,000 words
        slop_density = (len(findings) / total_words) * 1000
        # 0 to 10 scale normalized to 0-100%
        slop_score = min(100.0, round(slop_density * 8.5, 1))

    return findings, total_words, slop_score

def print_report(filepath, findings, total_words, slop_score):
    print("=" * 72)
    print("🧹 DESLOPIFY: AI Writing & Slop Linter Report")
    print("=" * 72)
    print(f"📄 Target File : {filepath}")
    print(f"📊 Total Words : {total_words:,}")
    print(f"🔍 Flags Found : {len(findings)}")
    
    if slop_score < 15.0:
        rating = "🟢 CLEAN / NATURAL (Low AI tells)"
    elif slop_score < 40.0:
        rating = "🟡 MODERATE SLOP (Noticeable AI patterns detected)"
    else:
        rating = "🔴 HEAVY SLOP (High concentration of formulaic LLM tropes)"
        
    print(f"📈 Slop Score  : {slop_score} / 100 [{rating}]")
    print("-" * 72)
    
    if not findings:
        print("✅ Outstanding! No prominent AI slop anti-patterns detected.")
        print("=" * 72)
        return
        
    print(f"{'Line':<6} | {'Offending Term / Pattern':<24} | {'Reason / Issue':<36}")
    print("-" * 72)
    for f in findings[:25]:
        term = f['term'][:22] + ".." if len(f['term']) > 22 else f['term']
        print(f"L{f['line']:<5} | {term:<24} | {f['description']:<36}")
        
    if len(findings) > 25:
        print(f"... and {len(findings) - 25} more instances.")
    print("=" * 72)
    print("💡 Tip: Use the 'deslopify' agent skill to automatically rewrite these passages.")

def main():
    parser = argparse.ArgumentParser(description="Deslopify: Scan and clean AI slop from text and markdown files.")
    parser.add_argument("path", help="Path to text or markdown file to inspect")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"❌ Error: File '{args.path}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    findings, total_words, slop_score = analyze_text(content)
    print_report(args.path, findings, total_words, slop_score)

if __name__ == "__main__":
    main()
