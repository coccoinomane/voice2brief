import os
from pathlib import Path
import openai
from pydub import AudioSegment # type: ignore
import tempfile
from anthropic import Anthropic
import argparse
from dotenv import load_dotenv

class AudioProcessor:
    def __init__(self, openai_api_key, anthropic_api_key=None):
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        openai.api_key = openai_api_key
        self.anthropic = Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None
        
        # Create transcripts directory if it doesn't exist
        Path("transcript").mkdir(exist_ok=True)

    def get_transcript_path(self, audio_file: str) -> Path:
        """Generate a unique path for the transcript file."""
        audio_path = Path(audio_file)
        return Path("transcript") / f"{audio_path.stem}_transcript.txt"

    def convert_audio_to_mp3(self, input_file: str) -> str:
        """Convert audio file to MP3 format if needed."""
        audio = AudioSegment.from_file(input_file)
        
        # Create temporary file for MP3
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_path = temp_file.name
            audio.export(temp_path, format='mp3')
        
        return temp_path

    def transcribe_audio(self, audio_file: str) -> str:
        """Transcribe audio file using OpenAI Whisper or return cached transcript."""
        transcript_path = self.get_transcript_path(audio_file)
        
        # Check if transcript already exists
        if transcript_path.exists():
            print(f"Using cached transcript {transcript_path}")
            return transcript_path.read_text()
            
        # Convert to MP3 if needed
        mp3_path = self.convert_audio_to_mp3(audio_file)
        
        try:
            with open(mp3_path, 'rb') as audio:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio
                )
            
            # Save transcript for future use
            transcript_path.write_text(transcript.text)
            return transcript.text
            
        finally:
            # Clean up temporary file
            os.unlink(mp3_path)

    def generate_brief(self, text: str, model: str = "gpt-4") -> dict:
        """Generate a team brief from transcribed text using specified model."""
        prompt = """
        Based on the following transcribed audio, create a clear team brief
        in markdown format.
        
        Generate the response in the same language as the transcription.
        
        Include:
        1. A concise overview of the topic
        2. Clear tasks assigned to team members
        3. Any deadlines or important dates mentioned
        
        Format the output as:
        
        OVERVIEW:
        [Overview text]
        
        ASSIGNMENTS:
        [List of assignments with clear ownership]
        
        TIMELINE:
        [Any deadlines or important dates]
        
        Transcription:
        {text}
        """
        
        if model.startswith("claude"):
            response = self.anthropic.messages.create(
                model=model,
                max_tokens=10000,
                messages=[{
                    "role": "user",
                    "content": prompt.format(text=text)
                }]
            )
            content = response.content[0].text
        else:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt.format(text=text)}
                ]
            )
            content = response.choices[0].message.content
            
        return {
            "type": "brief",
            "content": content
        }

    def generate_meeting_notes(self, text: str, model: str = "gpt-4") -> dict:
        """Generate meeting notes from transcribed text using specified model."""
        prompt = """
        Based on the following meeting transcription, create comprehensive meeting
        notes in markdown format.
        
        Generate the response in the same language as the transcription.
        
        Include:
        1. Meeting summary
        2. Key discussion points
        3. Action items with clear ownership
        4. Any decisions made
        
        Format the output as:
        
        SUMMARY:
        [Brief meeting summary]
        
        KEY POINTS DISCUSSED:
        [Bullet points of main discussion topics]
        
        ACTION ITEMS:
        [List of action items with owners]
        
        DECISIONS:
        [List of decisions made]
        
        Transcription:
        {text}
        """
        
        if model.startswith("claude"):
            response = self.anthropic.messages.create(
                model=model,
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": prompt.format(text=text)
                }]
            )
            content = response.content[0].text
        else:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt.format(text=text)}
                ]
            )
            content = response.choices[0].message.content
            
        return {
            "type": "meeting_notes",
            "content": content
        }

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process audio files and generate briefs or meeting notes.')
    
    parser.add_argument('audio_file', type=str, help='Path to the audio file')
    
    parser.add_argument('--mode', type=str, 
                       choices=['brief', 'meeting_notes'],
                       default='brief',
                       help='Processing mode: brief or meeting_notes')
    
    parser.add_argument('--model', type=str,
                       default='chatgpt-4o-latest',
                       help='LLM model to use for processing')
    
    parser.add_argument('--output', type=str,
                       help='Output file path (optional, defaults to stdout)')
    
    return parser.parse_args()

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API keys from environment
    openai_api_key = os.getenv('OPENAI_API_KEY')
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if we need Anthropic API key
    if args.model.startswith('claude') and not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables (required for Claude model)")
    
    # Initialize processor
    processor = AudioProcessor(openai_api_key, anthropic_api_key)
    
    try:
        # Step 1: Transcribe audio
        print("Transcribing audio with Whisper...")
        transcription = processor.transcribe_audio(args.audio_file)
        
        # Step 2: Generate output
        print(f"\nGenerating output with {args.model}...")
        
        if args.mode == "brief":
            result = processor.generate_brief(transcription, args.model)
        else:
            result = processor.generate_meeting_notes(transcription, args.model)
        
        # Handle output
        if args.output:
            with open(args.output, 'w') as f:
                f.write(result["content"])
            print(f"\nOutput written to: {args.output}")
        else:
            print("\nOutput:")
            print(result["content"])
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())