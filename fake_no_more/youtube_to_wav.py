import os
import yt_dlp as youtube_dl
from pydub import AudioSegment

def download_youtube_audio_as_wav(url, output_filename, target_directory):
    """
    Downloads the audio from a YouTube video and saves it as a WAV file in the specified directory.

    Args:
        url (str): The YouTube URL.
        output_filename (str): The name of the output WAV file.
        target_directory (str): The directory where the file will be saved.

    Returns:
        str: Path to the saved WAV file.
    """
    # Check if the directory exists, raise an error if it does not
    if not os.path.exists(target_directory):
        raise FileNotFoundError(f"The specified directory '{target_directory}' does not exist. Please provide a valid directory.")

    print(f"Saving file in: {target_directory}")

    # Download only audio using yt-dlp (no video stream merging)
    print(f"Downloading YouTube audio from {url}...")
    ydl_opts = {
        'format': 'bestaudio/best',  # Best audio format
        'outtmpl': os.path.join(target_directory, 'temp_audio.%(ext)s'),  # Save as temporary audio file in the target directory
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        audio_file_path = os.path.join(target_directory, f"temp_audio.{info_dict['ext']}")
        print(f"Downloaded audio to {audio_file_path}")

    # Convert the audio to WAV using pydub
    print("Converting audio to WAV...")
    audio = AudioSegment.from_file(audio_file_path)
    wav_file_path = os.path.join(target_directory, output_filename)
    audio.export(wav_file_path, format="wav")  # Save as WAV
    print(f"Audio successfully saved as WAV: {wav_file_path}")

    # Remove the temporary audio file
    os.remove(audio_file_path)

    return wav_file_path


def process_multiple_videos(urls, target_directory="../raw_data/demo_files"):
    """
    Processes multiple YouTube video URLs to download and convert them to WAV.

    Args:
        urls (list): A list of YouTube video URLs.
        target_directory (str): The directory where the audio files will be saved.

    Returns:
        list: Paths to the saved WAV files.
    """
    wav_files = []
    for i, url in enumerate(urls, start=1):
        output_filename = f"output_video_{i}.wav"  # Assign a unique filename for each video
        wav_file_path = download_youtube_audio_as_wav(url, output_filename, target_directory)
        wav_files.append(wav_file_path)
    return wav_files


# Example usage
if __name__ == "__main__":
    youtube_urls = [
        "https://www.youtube.com/watch?v=cQ54GDm1eL0",
        "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
        # Add more YouTube URLs here
    ]
    
    try:
        wav_files = process_multiple_videos(youtube_urls)
        print(f"Converted files: {wav_files}")
    except FileNotFoundError as e:
        print(e)  # Handle the error if the directory does not exist
