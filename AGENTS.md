# AEGIS-AD — Antigravity Project Instructions

## 1. Project Overview

**Project:** AEGIS-AD — Preventive AI Safety System  
**Repository:** https://github.com/kunalsahwb-tech/testing.git  
**Target Platform:** Python 3.11 with standard `src` layout  

AEGIS-AD is an AI-assisted safety and investigation system designed to detect defined safety events, track people and vehicles across camera views, raise alerts, and retrieve relevant video evidence.

Development is conducted in strictly controlled stages. Each stage must be verified and approved before proceeding to the next.

## 2. Permanent Working Rules

Follow these rules for all engineering and maintenance work in this repository:

1. **Use loop engineering for every change:** Follow the continuous cycle of *Inspect → Plan one small change → Implement → Run checks → Inspect output → Fix failures → Run checks again → Report verified result*.
2. **Inspect before editing:** Always read the current code, configuration, callers, and tests before making modifications.
3. **Test after each small change:** Run relevant tests and validation commands after every change rather than batching unrelated changes.
4. **Keep changes focused and easy to review:** Make small, clear, single-purpose modifications.
5. **Use simple, natural English in documentation:** Avoid marketing buzzwords or declaring unfinished features complete.
6. **Do not invent test results or performance claims:** Report only what commands actually produce. If a test was not run, say it was not run.
7. **Do not copy external repository code without approval:** Adapt components only when explicitly authorized.
8. **Complete and verify one stage before beginning another:** Strict stage gates govern development. Do not start a new stage until the prior stage is approved.
9. **Do not commit or push without explicit approval:** Never run destructive Git commands, commit, or push without the user's explicit consent.
10. **Do not add advanced AI features before the core workflow is working:** Features like cross-camera Re-ID, license plate recognition (LPR), and tool-calling AI investigators are scheduled for later stages once the core ingestion, detection, and tracking pipeline is proven.
