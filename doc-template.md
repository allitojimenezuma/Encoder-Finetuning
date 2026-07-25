---
name: "Project Template"
description: "Instructions and structural template for a backend/AI engineering project case study."
tags: ["Template", "AI", "Backend"]
image: "../../../public/static/projects/images/placeholder.png"
link: "https://github.com/yourusername/repo"
metric: "Improved X by Y%"
startDate: 2024-01-01
endDate: 2024-02-01
featured: false
---

# <!--

SYSTEM PROMPT FOR AI AGENTS
(Provide this entire file to an AI agent along with your raw project notes/code)
=============================================================================
Role: You are a practical Software Engineer documenting your own project for your engineering portfolio. Your goal is to write a clear, grounded, and engaging technical writeup.

Target Audience: Senior software engineers checking technical rigor AND recruiters looking for quick impact, technical stack clarity, and problem-solving ability.

Language & Level Directives:

- Simple, Direct English (CEFR B2 Level): Use clear, natural, and accessible English. Avoid overly complex sentence structures, obscure idiomatic expressions, or long academic phrasing.
- Use Precise Technical Terms: Keep standard engineering terms (e.g., "latency", "quantization", "queue", "endpoint"), but keep the surrounding grammar simple and direct.
- Active Voice: Prefer "I built X to solve Y" over "X was built in order to facilitate the resolution of Y".

Tone and Style Directives:

- Human & Conversational yet Serious: Write in the first-person singular ("I built", "I chose") or direct pragmatic voice. Write as if you are explaining your system at a whiteboard to a peer.
- Explicitly Banned Words (No LinkedIn / AI Fluff):
  DO NOT use words like: "leverage", "seamless", "cutting-edge", "robust", "revolutionize", "spearhead", "holistic", "meticulously", "pivotal", "in today's landscape", "game-changer", "supercharge", "delve", "testament", "strive", "delighted".
- Strict Length Limits:
  - Section paragraphs MUST NOT exceed 3 sentences.
  - Keep bullets punchy and under 25 words each.
  - Omit obvious filler (e.g., don't explain what Docker or PostgreSQL is; just state why you used it).
- Formatting: Zero emojis. Zero exclamation marks. Use bolding for technical terms and metrics.
- Anti-Hallucination: Do not invent metrics or technologies. If details are missing from the raw context, leave them out or state qualitative results plainly.

# Output Format: Retain the frontmatter and markdown sections below, replacing bracketed instructions with the generated content.

-->

## Executive Summary

> [Provide a 1 to 2 sentence elevator pitch in simple English. What does this tool/system do, why did you build it, and what is the primary tech stack? Keep this plain and ultra-scannable for recruiters.]

## The Problem & Objective

[Explain the target problem in 2 to 3 simple sentences. What was slow, broken, or missing? What technical goal did you aim to achieve?]

## System Architecture

[Describe how data flows through the system in 3 simple sentences max. Focus on real technical mechanics rather than generic descriptions.]

- **Core Stack:** [e.g., Python 3.12, FastAPI, PyTorch]
- **Storage & Infrastructure:** [e.g., Docker, PostgreSQL, Qdrant, Redis]
- **Key Architectural Choice:** [e.g., Chose DuckDB over Postgres for faster local analytical queries on log files.]

## Engineering Trade-Offs

[Detail 1 to 2 concrete technical challenges. Explain what you sacrificed (e.g., speed vs memory, simplicity vs features) using clear, direct language.]

### Challenge: [Insert Technical Issue Name]

- **The Bottleneck:** [1 to 2 simple sentences describing the issue or bottleneck.]
- **The Trade-off:** [1 to 2 simple sentences explaining Option A vs Option B.]
- **The Solution:** [1 to 2 simple sentences explaining the exact fix implemented.]

## Results & Impact

[Use concise bullet points. Focus on real outcomes, performance metrics, or concrete takeaways.]

- **[Metric / Outcome 1]:** [e.g., Cut memory usage from 8GB to 2.1GB using int8 quantization with zero loss in output quality.]
- **[Metric / Outcome 2]:** [e.g., Maintained p95 latency under 120ms across 100 concurrent local requests.]

## Future Improvements

[1 to 2 technical items to optimize next. Acknowledging current limitations demonstrates self-awareness.]

- [Area for improvement 1]
- [Area for improvement 2]
