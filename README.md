# Voice to Brief

Convert voice memos into actionable briefs and meeting notes using AI transcription and processing.

This tool helps you transform audio recordings (like iOS Voice Memos) into well-structured written content using OpenAI's Whisper for transcription and large language models (GPT-4o, o1, o3-mini or Claude) for processing. Perfect for busy professionals who prefer voice recording over typing.

## Features

- **Audio Transcription**: Convert any audio file to text using OpenAI's Whisper model
- **Two Processing Modes**:
  - **Brief Generation**: Turn voice memos about tasks into clear team briefs with assignments
  - **Meeting Notes**: Convert recorded meetings into structured notes with action items
- **Multiple LLM Support**: Compatible with OpenAI's GPT and Anthropic's Claude
- **Flexible Output**: Display results in console or save to file

## Installation

1. Clone the repository:
```bash
git clone https://github.com/coccoinomane/voice2brief.git
cd voice2brief
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Copy the `.env.example` file to `.env` and add your API keys:
```bash
OPENAI_API_KEY=your-openai-api-key
# Optional, only needed for Claude
ANTHROPIC_API_KEY=your-anthropic-api-key
```

## Usage

Basic usage:
```bash
python voice2brief.py path/to/audio.m4a
```

Command Line Arguments:
- `audio_file`: Path to the audio file (required)
- `--mode`: Processing mode (default: brief)
  - `brief`: Generate a team brief with assignments
  - `meeting_notes`: Generate meeting notes with action items
- `--model`: LLM model to use (default: gpt-4o-latest)
- `--output`: Output file path (optional, defaults to stdout)

You can choose any model supported by OpenAI or Anthropic.  Examples:
```bash
# Use the latest GPT-4o model
python voice2brief.py audio.m4a --model chatgpt-4o-latest

# Use the o1 reasoning model
python voice2brief.py audio.m4a --model o1-2024-12-17

# Use the o3-mini reasoning model
python voice2brief.py audio.m4a --model o3-mini-2025-01-31

# Use the Claude Sonnet model
python voice2brief.py audio.m4a --model claude-3-sonnet-20241022
```

Please note that for OpenAI models, you might need a certain tier of account to access more advanced models like o1 or o3.

## Example Outputs

### Brief Mode
```
OVERVIEW:
[A concise overview of the topic]

ASSIGNMENTS:
[List of assignments with clear ownership]

TIMELINE:
[Any deadlines or important dates]
```

### Meeting Notes Mode
```
SUMMARY:
[Brief meeting summary]

KEY POINTS DISCUSSED:
[Bullet points of main discussion topics]

ACTION ITEMS:
[List of action items with owners]

DECISIONS:
[List of decisions made]
```

## Requirements

- Python 3.10+
- OpenAI API key (required)
- Anthropic API key (optional, only for Claude)

## Contributing

Feel free to open issues or submit pull requests to improve the tool.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.