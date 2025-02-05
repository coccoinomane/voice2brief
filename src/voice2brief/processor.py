"""Core functionality for processing audio files and generating structured text."""

import os
from pathlib import Path
import tempfile
from typing import Literal, TypedDict, Optional

import openai
from anthropic import Anthropic
from pydub import AudioSegment  # type: ignore

OutputType = Literal["brief", "meeting_notes", "extended"]


class OutputResult(TypedDict):
    """Type definition for the output result."""

    type: OutputType
    content: str


class AudioProcessor:
    """Process audio files and generate structured text using AI models."""

    def __init__(
        self, openai_api_key: str, anthropic_api_key: Optional[str] = None
    ) -> None:
        """Initialize the AudioProcessor.

        Args:
            openai_api_key: OpenAI API key for transcription and text generation
            anthropic_api_key: Optional Anthropic API key for Claude models
        """
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        openai.api_key = openai_api_key
        self.anthropic = (
            Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None
        )

        # Create transcripts directory if it doesn't exist
        Path("transcript").mkdir(exist_ok=True)

    def get_transcript_path(self, audio_file: str) -> Path:
        """Generate a unique path for the transcript file.

        Args:
            audio_file: Path to the audio file

        Returns:
            Path object for the transcript file
        """
        audio_path = Path(audio_file)
        return Path("transcript") / f"{audio_path.stem}_transcript.txt"

    def convert_audio_to_mp3(self, input_file: str) -> str:
        """Convert audio file to MP3 format if needed.

        Args:
            input_file: Path to the input audio file

        Returns:
            Path to the temporary MP3 file
        """
        audio = AudioSegment.from_file(input_file)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name
            audio.export(temp_path, format="mp3")

        return temp_path

    def transcribe_audio(self, audio_file: str) -> str:
        """Transcribe audio file using OpenAI Whisper or return cached transcript.

        Args:
            audio_file: Path to the audio file

        Returns:
            Transcribed text
        """
        transcript_path = self.get_transcript_path(audio_file)

        if transcript_path.exists():
            return transcript_path.read_text()

        mp3_path = self.convert_audio_to_mp3(audio_file)

        try:
            with open(mp3_path, "rb") as audio:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1", file=audio
                )

            transcript_path.write_text(transcript.text)
            return transcript.text

        finally:
            os.unlink(mp3_path)

    def generate_brief(self, text: str, model: str = "gpt-4") -> OutputResult:
        """Generate a team brief from transcribed text.

        Args:
            text: Transcribed text to process
            model: AI model to use for generation

        Returns:
            Dictionary containing the brief content
        """
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

        content = self._generate_content(prompt, text, model)
        return {"type": "brief", "content": content}

    def generate_meeting_notes(self, text: str, model: str = "gpt-4") -> OutputResult:
        """Generate meeting notes from transcribed text.

        Args:
            text: Transcribed text to process
            model: AI model to use for generation

        Returns:
            Dictionary containing the meeting notes
        """
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

        content = self._generate_content(prompt, text, model)
        return {"type": "meeting_notes", "content": content}

    def generate_extended(self, text: str, model: str = "gpt-4") -> OutputResult:
        """Transform transcribed text into a polished document.

        Args:
            text: Transcribed text to process
            model: AI model to use for generation

        Returns:
            Dictionary containing the extended content
        """
        prompt = """
        Based on the following transcribed audio, create a clear, well-structured text
        that preserves the speaker's intent and message. This should read as a proper
        document rather than a transcript.
        
        Generate the response in the same language as the transcription.
        
        Guidelines:
        1. Maintain the original message and key points
        2. Improve clarity and flow
        3. Fix any verbal artifacts or repetitions
        4. Organize the content logically
        5. Keep the speaker's tone and style
        
        Transcription:
        {text}
        """

        content = self._generate_content(prompt, text, model)
        return {"type": "extended", "content": content}

    def _generate_content(self, prompt: str, text: str, model: str) -> str:
        """Generate content using the specified AI model.

        Args:
            prompt: Template prompt for the generation
            text: Text to process
            model: AI model to use

        Returns:
            Generated content
        """
        if model.startswith("claude"):
            if not self.anthropic:
                raise ValueError("Anthropic API key required for Claude models")

            response = self.anthropic.messages.create(
                model=model,
                max_tokens=10000,
                messages=[{"role": "user", "content": prompt.format(text=text)}],
            )
            return response.content[0].text
        else:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt.format(text=text)}],
            )
            return response.choices[0].message.content
