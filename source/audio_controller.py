import numpy as np
import sounddevice as sd
import soundfile as sf
import pyrubberband as pyrb
from collections import deque
from constants import Command, AudioRegion
import os
import librosa
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torch
import tempfile
import warnings
warnings.filterwarnings('ignore')

class AudioTrack:
    """개별 오디오 트랙 관리"""
    def __init__(self, audio_file_path, initial_tempo=90.0):
        self.audio_file_path = audio_file_path
        self.audio_data, self.sample_rate = sf.read(audio_file_path)
        self.position = 0.0
        self.is_playing = False
        self.volume = 0.0
        self.original_volume = 1.0
        
        self.min_volume = 0.05
        self.max_volume = 1.5
        self.target_volume = self.original_volume
        self.smoothing_factor = 0.2
        
        if len(self.audio_data.shape) == 1:
            self.audio_data = np.column_stack((self.audio_data, self.audio_data))
        
        # 템포 관련 속성 초기화
        self.target_tempo = initial_tempo
        self.current_tempo = initial_tempo
        self.tempo_smoothing_factor = 1.0
        print(f"Initial tempo: {initial_tempo}")
        
        # 템포별 오디오 데이터 미리 계산
        self.tempo_variants = {}
        self._precompute_tempo_variants()
        
        # current_audio 초기화
        self.current_audio = self.tempo_variants[initial_tempo]
        self.current_rms = 0.0
        
    def _precompute_tempo_variants(self):
        """템포별 오디오 데이터 미리 계산 (0.5배~1.5배, 0.1 단위)"""
        print(f"Precomputing tempo variants for {self.audio_data.shape}")
        base_tempo = self.current_tempo
        tempo_ratios = np.arange(0.5, 1.55, 0.1)
        
        # 캐시 디렉토리 생성
        cache_dir = "audio_cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # base_tempo(90.0)에 대한 데이터 처리
        self.tempo_variants[base_tempo] = self.audio_data.copy()
        
        # 나머지 템포 처리
        for ratio in tempo_ratios:
            if ratio != 1.0:  # base_tempo는 이미 처리했으므로 건너뜀
                tempo = base_tempo * ratio
                # 각 스템별로 구분된 캐시 파일명 사용
                stem_name = os.path.basename(self.audio_file_path).split('.')[0]  # 파일명에서 스템 이름 추출
                cache_file = os.path.join(cache_dir, f"{stem_name}_tempo_{tempo:.1f}.npy")
                
                if os.path.exists(cache_file):
                    # 캐시된 파일이 있으면 로드
                    print(f"Loading cached tempo {tempo:.1f} BPM for {stem_name}")
                    self.tempo_variants[tempo] = np.load(cache_file)
                else:
                    # 캐시된 파일이 없으면 새로 계산하고 저장
                    print(f"Computing tempo {tempo:.1f} BPM for {stem_name} (ratio: {ratio:.2f})")
                    stretched_audio = pyrb.time_stretch(self.audio_data, self.sample_rate, ratio)
                    self.tempo_variants[tempo] = stretched_audio
                    np.save(cache_file, stretched_audio)
        
    def set_tempo(self, target_bpm):
        """BPM을 넘으면 해당 템포로 변경"""
        available_tempos = sorted(list(self.tempo_variants.keys()))
        
        # 목표 템포 설정
        self.target_tempo = target_bpm
        
        # EMA 적용
        self.current_tempo = (self.tempo_smoothing_factor * self.target_tempo + 
                            (1 - self.tempo_smoothing_factor) * self.current_tempo)
        
        # 현재 템포보다 큰 것 중 가장 작은 포 선택
        for tempo in available_tempos:
            if tempo >= self.current_tempo:
                self.current_audio = self.tempo_variants[tempo]
                break
        else:  # 모든 템포보다 크면 가장 큰 템포 선택
            self.current_audio = self.tempo_variants[available_tempos[-1]]
        
    def set_volume(self, volume):
        """면적 기반 볼륨 설정 (기존 코드 유지)"""
        self.target_volume = self.original_volume * (self.min_volume + volume * (self.max_volume - self.min_volume))
        
        self.volume = (self.smoothing_factor * self.target_volume + 
                      (1 - self.smoothing_factor) * self.volume)
     
    def get_next_frames(self, num_frames):
        """다음 프레임 가져오기"""
        current_frame = int(self.position * len(self.current_audio))
        end_frame = current_frame + num_frames
        
        if end_frame > len(self.current_audio):
            frames = np.zeros((num_frames, 2))
            remaining = len(self.current_audio) - current_frame
            frames[:remaining] = self.current_audio[current_frame:]
            frames[remaining:] = self.current_audio[:end_frame - len(self.current_audio)]
            self.position = (end_frame - len(self.current_audio)) / len(self.current_audio)
        else:
            frames = self.current_audio[current_frame:end_frame]
            self.position = end_frame / len(self.current_audio)
        
        # 볼륨 적용 전의 원본 프레임으로 RMS 계산
        self.current_rms = np.sqrt(np.mean(frames**2))
        
        # 볼륨 적용된 프레임 반환
        return frames * self.volume

class AudioController:
    def __init__(self, audio_file_path=None):
        self.current_tempo = 90.0
        self.active_solos = set()
        self.tracks = {}
        
        if audio_file_path:
            self.initialize_from_audio(audio_file_path)
            
        # 오디오 스트림 설정
        if self.tracks:
            self.stream = sd.OutputStream(
                samplerate=next(iter(self.tracks.values())).sample_rate,
                channels=2,
                callback=self.audio_callback,
                blocksize=4096 * 4,
                latency='high'
            )
            self.stream.start()
            
    def initialize_from_audio(self, audio_file_path):
        """오디오 파일로부터 stem 분리 및 BPM 감지"""
        print("Loading audio file and initializing stems...")
        
        # 1. BPM 감지
        y, sr = librosa.load(audio_file_path)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        self.current_tempo = float(tempo)
        print(f"Detected tempo: {self.current_tempo} BPM")
        
        # 2. Stem 파일 확인
        temp_dir = "temp_stems"
        os.makedirs(temp_dir, exist_ok=True)
        
        stem_mapping = {
            "drums": AudioRegion.DRUM,
            "bass": AudioRegion.BASS,
            "other": AudioRegion.MELODY,  # melody로 사용
            "vocals": AudioRegion.VOCAL
        }
        
        # 모든 stem 파일이 있는지 확인
        all_stems_exist = all(
            os.path.exists(os.path.join(temp_dir, f"{stem_name}.wav"))
            for stem_name in stem_mapping.keys()
        )
        
        if all_stems_exist:
            print("Found existing stem files, skipping separation...")
            # 기존 stem 파일들 사용
            for stem_name, region in stem_mapping.items():
                stem_path = os.path.join(temp_dir, f"{stem_name}.wav")
                self.tracks[region] = AudioTrack(stem_path, initial_tempo=self.current_tempo)
        else:
            print("Separating stems...")
            # Stem 분리 수행
            model = get_model('htdemucs')
            model.cpu()
            wav, sr = librosa.load(audio_file_path, sr=44100, mono=False)
            if wav.ndim == 1:
                wav = wav[None]
            wav = torch.tensor(wav)[None]
            
            stems = apply_model(model, wav, progress=True)[0]
            stems = stems.cpu().numpy()
            
            # stem 파일 저장 및 AudioTrack 초기화
            stem_idx_mapping = {
                0: "drums",
                1: "bass",
                2: "other",
                3: "vocals"
            }
            
            for idx, stem_name in stem_idx_mapping.items():
                stem_path = os.path.join(temp_dir, f"{stem_name}.wav")
                sf.write(stem_path, stems[idx].T, sr)
                region = stem_mapping[stem_name]
                self.tracks[region] = AudioTrack(stem_path, initial_tempo=self.current_tempo)
        
        print("Stem initialization complete!")
        
    def sync_all_tracks(self):
        """모든 트랙의 위치 동기화"""
        for track in self.tracks.values():
            track.position = 0.0

    def set_gui_manager(self, gui_manager):
        """GUI 매니저 설정"""
        self.gui_manager = gui_manager
        
        # GUI 초기 상태 설정
        for region in AudioRegion.get_all_regions():
            if region != AudioRegion.MASTER:
                self.gui_manager.update_track_info(
                    region,
                    volume=self.tracks[region].volume if region in self.tracks else 0.0,
                    conducting=False,
                    soloed=False
                )

    def audio_callback(self, outdata, frames, time, status):
        """오디오 스트림 콜백"""
        if status:
            print(f'Audio callback status: {status}')
                
        mixed_output = np.zeros((frames, 2))
        
        # 각 트랙의 오디오 데이터 처리 및 시각화 업데이트
        for region, track in self.tracks.items():
            frames_data = track.get_next_frames(frames)
            mixed_output += frames_data
            
            # GUI가 설정되어 있으면 시각화 업데이트 (원본 RMS 값도 전달)
            if hasattr(self, 'gui_manager'):
                self.gui_manager.visualizer.update_audio_data(
                    region, 
                    frames_data,
                    original_rms=track.current_rms  # 본 RMS 값 전달
                )
                
        # 클리핑 방지
        outdata[:] = np.clip(mixed_output, -1.0, 1.0)

    def execute_command(self, command):
        """명령어 실행"""
        # TEMPO 명령 처리를 가장 먼저 수행
        if "TEMPO" in command and command["TEMPO"] > 0:
            target_bpm = command["TEMPO"]
            self.current_tempo = target_bpm
            # 모든 트랙의 템포를 ���시에 변경
            current_position = next(iter(self.tracks.values())).position  # 현재 위치 저장
            for track in self.tracks.values():
                track.set_tempo(target_bpm)
                track.position = current_position  # 위치 동기화 유지
        
        # SOLO 명령어 처리
        solo_commands = {
            Command.MELODY_SOLO: AudioRegion.MELODY,
            Command.VOCAL_SOLO: AudioRegion.VOCAL,
            Command.BASS_SOLO: AudioRegion.BASS,
            Command.DRUM_SOLO: AudioRegion.DRUM
        }
        
        # 1. SOLO OFF 상태 명시적 체크
        solo_off = True
        for cmd in solo_commands:
            if command.get(cmd, 0) > 0:
                solo_off = False
                break
        
        # 2. SOLO OFF 처리 (최우선)
        if solo_off and self.active_solos:
            print("SOLO OFF detected")  # 디버깅용
            for region, track in self.tracks.items():
                if track.pre_solo_volume is not None:
                    track.volume = track.pre_solo_volume
                    track.pre_solo_volume = None
            self.active_solos.clear()
            
            # GUI 업데이트
            if self.gui_manager:
                for region in AudioRegion.get_all_regions():
                    if region != AudioRegion.MASTER:
                        self.gui_manager.update_track_info(
                            region,
                            volume=self.tracks[region].volume,
                            soloed=False
                        )
            return  # SOLO OFF 처리 후 로 리턴
        
        # 3. SOLO ON 처리
        for cmd, confidence in command.items():
            if cmd in solo_commands and confidence > 0:
                region = solo_commands[cmd]
                
                # 처음 SOLO 시작 시 볼륨 저장
                if not self.active_solos:
                    for r, track in self.tracks.items():
                        track.pre_solo_volume = track.volume
                
                # 볼륨 설정
                for r, track in self.tracks.items():
                    track.volume = track.original_volume if r == region else 0.0
                
                self.active_solos = {region}
                
                # GUI 업데이트
                if self.gui_manager:
                    for r in AudioRegion.get_all_regions():
                        if r != AudioRegion.MASTER:
                            self.gui_manager.update_track_info(
                                r,
                                volume=self.tracks[r].volume,
                                soloed=(r == region)
                            )
                break  # 한 번에 하나의 SOLO만 처리
        
        # 나머지 명령어 처리 (CONDUCT, PLAY/STOP 등)
        # CONDUCT 명령어 처리
        for region in AudioRegion.get_all_regions():
            if region != AudioRegion.MASTER:
                if command.get(f"{region}_CONDUCT", 0) > 0:
                    volume = command.get(f"{region}_CONDUCT_VOLUME", 0.0)
                    bpm = command.get(f"{region}_CONDUCT_BPM")
                    
                    self.tracks[region].set_volume(volume)
                    if bpm is not None:
                        self.tracks[region].set_tempo(bpm)
                        if self.gui_manager:
                            self.gui_manager.update_bpm(bpm)
                    
                    if self.gui_manager:
                        self.gui_manager.update_track_info(
                            region, 
                            volume=self.tracks[region].volume,
                            conducting=True
                        )
                    
        # PLAY/STOP 명령어 처리
        command_region_map = {
            Command.BASS_PLAY: (AudioRegion.BASS, True),
            Command.BASS_STOP: (AudioRegion.BASS, False),
            Command.VOCAL_PLAY: (AudioRegion.VOCAL, True),
            Command.VOCAL_STOP: (AudioRegion.VOCAL, False),
            Command.MELODY_PLAY: (AudioRegion.MELODY, True),
            Command.MELODY_STOP: (AudioRegion.MELODY, False),
            Command.DRUM_PLAY: (AudioRegion.DRUM, True),
            Command.DRUM_STOP: (AudioRegion.DRUM, False)
        }
        
        for cmd, confidence in command.items():
            if confidence > 0 and cmd in command_region_map:
                region, should_play = command_region_map[cmd]
                if region in self.tracks:
                    track = self.tracks[region]
                    if should_play:
                        track.volume = track.original_volume
                    else:
                        track.volume = 0.0

    def start_playback(self):
        """모든 트랙의 재생을 동시에 시작"""
        self.sync_all_tracks()  # 재생 시작 전 모든 트랙 동기화
        for track in self.tracks.values():
            track.is_playing = True

    def __del__(self):
        self.stream.stop()
        self.stream.close() 