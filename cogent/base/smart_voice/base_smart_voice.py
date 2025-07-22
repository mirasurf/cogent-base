from abc import ABC, abstractmethod


class SmartVoiceBase(ABC):
    @abstractmethod
    def transcribe(self, audio_data):
        """Transcribe audio data to text."""
