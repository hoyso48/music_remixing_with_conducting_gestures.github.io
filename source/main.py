import cv2
import time
import librosa
import soundfile as sf
from gui import GUIManager
from audio_controller import AudioController
from gesture_recognizer import GestureRecognizer
from constants import AudioRegion, Command
import os

def get_audio_file():
    """Get audio file input from user"""
    while True:
        print("\nEnter the name of the audio file (supports mp3, wav, flac):")
        filename = input().strip()
        
        # Check if file exists
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' not found.")
            continue
            
        # Check file extension
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ['.mp3', '.wav', '.flac']:
            print(f"Error: Unsupported file format. Only mp3, wav, flac are supported.")
            continue
            
        return filename

def main():
    cap = cv2.VideoCapture(0)
    
    # Select audio file
    audio_file = get_audio_file()
    
    # Initialize audio controller and detect initial BPM
    audio_controller = AudioController(audio_file)
    gui_manager = GUIManager()
    gesture_recognizer = GestureRecognizer()
    
    # Set initial BPM
    y, sr = librosa.load(audio_file)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    initial_bpm = float(tempo)
    gui_manager.update_bpm(initial_bpm)
    print(f"Detected initial BPM: {initial_bpm:.1f}")
    
    audio_controller.set_gui_manager(gui_manager)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        # 제스처 인식 결과 받기
        gesture_states, command = gesture_recognizer.process_frame(frame)
        
        # GUI 업데이트 요청
        gui_manager.update(frame, gesture_states, command)
        
        # 처리된 프레임 가져오기
        processed_frame = gui_manager.get_processed_frame()
        if processed_frame is not None:
            cv2.imshow('Hand Gesture Recognition', processed_frame)
        
        # 오디오 컨트롤러 업데이트
        if command.get(Command.NONE, 1.0) < 1:
            audio_controller.execute_command(command)
            # BPM이 변경되었다면 GUI 업데이트
            if command.get("TEMPO") is not None:
                gui_manager.update_bpm(command["TEMPO"])
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 