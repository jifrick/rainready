"""Central Gemini 4 + Decisions API model registry.

Follows the 'Build with Gemini 4: Hackathon Builder Guide'.
"""

MODEL_IDS: dict[str, str] = {
    # Gemini 4 reasoning and coding models (1M unified context + long decode)
    "frontier_deep": "gemini-4-carbon",
    "frontier_fast": "gemini-4-neon",
    "fast_workhorse": "gemini-4-hydrogen",
    "frontier_argon": "gemini-4-argon",
    "text_fast_lite": "gemini-3.5-flash-lite",
    # Calibrated multimodal routing, gating, and rubric scoring (v1beta)
    "decisions": "gemini-decision-1",
    # Managed agents (v1beta)
    "antigravity_agent": "antigravity-preview-09-2026",
    "deep_research": "deep-research-preview-04-2026",
    "deep_research_max": "deep-research-max-preview-04-2026",
    # Live voice, camera, transcription, translation, and speech
    "live_voice_vision": "gemini-3.8-live",
    "live_extended_thinking": "gemini-3.8-live-extended-thinking",
    "transcribe_batch": "gemini-3.5-transcribe",
    "transcribe_live": "gemini-3.5-transcribe-live",
    "live_translate": "gemini-3.5-live-translate-preview",
    "tts_studio": "gemini-3.8-flash-tts",
    "tts_fast": "gemini-3.8-flash-lite-tts",
    # Native images, video with audio, and music
    "image_default": "gemini-nano-banana-2.1",
    "image_pro": "gemini-3-pro-image",
    "image_fast": "gemini-3.1-flash-lite-image",
    "video_audio": "gemini-omni-1.2-flash",
    "music_song": "lyria-3.5",
    "music_loop": "lyria-3-clip-preview",
    # Open-weight Gemma 4 models on the Interactions API
    "gemma_dense": "gemma-4-31b-it",
    "gemma_moe": "gemma-4-26b-a4b-it",
}
