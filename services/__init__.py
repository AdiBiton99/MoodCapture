"""
services — external integrations (OpenAI, etc.).

This package is intentionally separate from `screen_emotion` so the existing
emotion-analysis pipeline has zero new dependencies. Modules here are imported
lazily and fail gracefully if optional packages are missing.
"""
