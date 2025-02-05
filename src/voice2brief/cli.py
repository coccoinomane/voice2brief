"""Command-line interface for voice2brief."""

import argparse
import os
import sys
from textwrap import dedent

from voice2brief.processor import AudioProcessor, OutputType


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.Reurns:
    Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Process audio files and generate briefs or meeting notes."
    )

    parser.add_argument("audio_file", type=str, help="Path to the audio file")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["brief", "meeting_notes", "extended"],
        default="brief",
        help="Processing mode: brief, meeting_notes, or extended",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="chatgpt-4o-latest",
        help="LLM model to use for processing",
    )

    parser.add_argument(
        "--output", type=str, help="Output file path (optional, defaults to stdout)"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code (0 for success, 1 for error)
    """

    # Get API keys from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_api_key:
        print(
            dedent(
                """
                Error: OPENAI_API_KEY not found, set it in your environment.
                On Linux or MacOS:
                    export OPENAI_API_KEY=your-openai-api-key
                On Windows (PowerShell):
                    setx OPENAI_API_KEY "your-openai-api-key
                """
            ).strip(),
            file=sys.stderr,
        )
        return 1

    # Parse command line arguments
    args = parse_arguments()

    # Check if we need Anthropic API key
    if args.model.startswith("claude") and not anthropic_api_key:
        print(
            dedent(
                """
                Error: ANTHROPIC_API_KEY not found, set it in your environment.
                On Linux or MacOS:
                    export ANTHROPIC_API_KEY=your-anthropic-api-key
                On Windows (PowerShell):
                    setx ANTHROPIC_API_KEY "your-anthropic-api-key
                """
            ).strip(),
            file=sys.stderr,
        )
        return 1

    # Initialize processor
    processor = AudioProcessor(openai_api_key, anthropic_api_key)

    try:
        # Step 1: Transcribe audio
        print("Transcribing audio with Whisper...", file=sys.stderr)
        transcription = processor.transcribe_audio(args.audio_file)

        # Step 2: Generate output
        print(f"\nGenerating output with {args.model}...", file=sys.stderr)

        mode: OutputType = args.mode  # type: ignore
        if mode == "brief":
            result = processor.generate_brief(transcription, args.model)
        elif mode == "meeting_notes":
            result = processor.generate_meeting_notes(transcription, args.model)
        else:  # extended mode
            result = processor.generate_extended(transcription, args.model)

        # Handle output
        if args.output:
            with open(args.output, "w") as f:
                f.write(result["content"])
            print(f"\nOutput written to: {args.output}", file=sys.stderr)
        else:
            print("\nOutput:", file=sys.stderr)
            print(result["content"])

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
