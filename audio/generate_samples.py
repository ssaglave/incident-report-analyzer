"""
Generate sample 911 emergency audio files for testing the audio pipeline.
Uses Google Text-to-Speech (gTTS) to create realistic emergency call samples.
"""

import os
from gtts import gTTS

SAMPLE_CALLS = [
    {
        "filename": "call_001_fire.mp3",
        "text": (
            "Hello, 911? There is a fire at 245 Main Street! "
            "People are trapped on the second floor. Please send help immediately! "
            "I can see thick black smoke coming from the windows. "
            "There are children inside, please hurry!"
        ),
    },
    {
        "filename": "call_002_accident.mp3",
        "text": (
            "Yes, I need to report a car accident on Highway 101 near Oak Avenue. "
            "Two vehicles collided head on. One driver appears to be unconscious. "
            "There is broken glass everywhere and one car is leaking fuel. "
            "We need an ambulance right away."
        ),
    },
    {
        "filename": "call_003_robbery.mp3",
        "text": (
            "I just witnessed a robbery at the First National Bank on Elm Street! "
            "Two men with masks ran out of the bank carrying bags. "
            "They got into a black SUV and drove north on Pine Road. "
            "I think they had guns. People inside are scared."
        ),
    },
    {
        "filename": "call_004_assault.mp3",
        "text": (
            "Please help, there is a fight at Central Park near the fountain. "
            "A man is attacking someone with a knife. "
            "The victim is bleeding and needs medical attention. "
            "The attacker is wearing a red jacket and ran towards the parking lot."
        ),
    },
    {
        "filename": "call_005_disturbance.mp3",
        "text": (
            "Hi, I want to report a noise complaint at 78 Maple Drive. "
            "My neighbors are having a loud party and it is past midnight. "
            "There seems to be a lot of people and some yelling. "
            "I have asked them to keep it down but they refused."
        ),
    },
]


def generate_samples(output_dir: str):
    """Generate sample audio files in the given directory."""
    os.makedirs(output_dir, exist_ok=True)

    for call in SAMPLE_CALLS:
        filepath = os.path.join(output_dir, call["filename"])
        if os.path.exists(filepath):
            print(f"  ✓ Already exists: {call['filename']}")
            continue

        print(f"  ⏳ Generating: {call['filename']}...")
        tts = gTTS(text=call["text"], lang="en", slow=False)
        tts.save(filepath)
        print(f"  ✓ Saved: {filepath}")

    print(f"\n✅ {len(SAMPLE_CALLS)} sample audio files ready in {output_dir}")


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    generate_samples(data_dir)
