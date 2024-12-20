import cv2
import numpy as np
from collections import deque
from scipy.fft import fft
from constants import AudioRegion, Command
import queue
import threading

class AudioVisualizer:
    def __init__(self, height=100, width=200):
        self.height = height
        self.width = width
        self.history_size = width
        
        # 각 영역별 색상 및 시각화 데이터
        self.visualizer_data = {
            "MELODY": {
                "waveform": deque(np.zeros(self.history_size), maxlen=self.history_size),
                "color": (255, 183, 77),  # 주황색
                "gradient": [(255, 183, 77), (255, 127, 0)]  # 밝은 주황 -> 진한 주황
            },
            "VOCAL": {
                "waveform": deque(np.zeros(self.history_size), maxlen=self.history_size),
                "color": (147, 112, 219),  # 보라색
                "gradient": [(187, 152, 255), (107, 72, 179)]  # 밝은 보라 -> 진한 보라
            },
            "BASS": {
                "waveform": deque(np.zeros(self.history_size), maxlen=self.history_size),
                "color": (65, 105, 225),  # 로얄 블루
                "gradient": [(105, 145, 255), (25, 65, 185)]  # 밝은 파랑 -> 진한 파랑
            },
            "DRUM": {
                "waveform": deque(np.zeros(self.history_size), maxlen=self.history_size),
                "color": (50, 205, 50),  # 라임 그린
                "gradient": [(90, 245, 90), (10, 165, 10)]  # 밝은 초록 -> 진한 초록
            }
        }

    def update_audio_data(self, region, audio_frames, original_rms=None):
        """오디오 데이터 업데이트"""
        if region not in self.visualizer_data:
            return
        
        # 더 작은 청크로 분할
        chunk_size = len(audio_frames) // 8
        chunks = np.array_split(audio_frames, 8)
        
        for chunk in chunks:
            waveform = np.mean(chunk[::8], axis=1)
            self.visualizer_data[region]["waveform"].extend(waveform)
            
        # 원본 RMS 값 저장 (볼륨 적용 전)
        if original_rms is not None:
            self.visualizer_data[region]["original_rms"] = original_rms

class GUIManager:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_thickness = 2
        self.font_color = (255, 255, 255)
        self.gesture_font_color = (0, 255, 0)
        self.grid_color = (128, 128, 128)
        self.hand_marker_color = (0, 255, 0)
        self.hand_marker_size = 10
        
        # 오디오 시각화 초기화
        self.visualizer = AudioVisualizer()
        
        # 각 영역의 텍스트 매핑
        self.region_texts = {
            AudioRegion.MELODY: "Melody",
            AudioRegion.VOCAL: "Vocal",
            AudioRegion.BASS: "Bass",
            AudioRegion.DRUM: "Drum"
        }
        
        # 초기 상태 설정
        self._states = {
            "MELODY": {"playback": "STOPPED", "conducting": False, "volume": 0.0},
            "VOCAL": {"playback": "STOPPED", "conducting": False, "volume": 0.0},
            "BASS": {"playback": "STOPPED", "conducting": False, "volume": 0.0},
            "DRUM": {"playback": "STOPPED", "conducting": False, "volume": 0.0}
        }
        
        self.prev_frame_time = 0
        self.fps = 0
        
        # RMS 막대 그래프 관련 속성 추가
        self.rms_bar_width = 20
        self.rms_bar_spacing = 10
        self.rms_bar_max_height = 100
        self.rms_bar_base_alpha = 0.3
        
        # 비동기 처리를 위한 속성 추가
        self.frame_buffer = queue.Queue(maxsize=3)
        self.processed_frame_buffer = queue.Queue(maxsize=3)
        self.running = True
        
        # BPM 표시를 위한 속성 추가
        self.current_bpm = 90.0
        self.bpm_color = (0, 255, 255)  # 노란색
        
        # GUI 처리 스레드 시작
        self.gui_thread = threading.Thread(target=self._process_gui)
        self.gui_thread.daemon = True
        self.gui_thread.start()
        
    def _process_gui(self):
        """GUI 처리 스레드"""
        while self.running:
            try:
                frame_data = self.frame_buffer.get(timeout=0.1)
                if frame_data is None:
                    continue
                    
                frame, gesture_states, command = frame_data
                processed_frame = frame.copy()
                
                # 기존 GUI 처리 로직 실행
                self.draw_grid(processed_frame)
                self.draw_section_texts(processed_frame)
                self.draw_command_states(processed_frame, command)
                self.draw_hand_position(processed_frame, gesture_states)
                self.draw_audio_visualization(processed_frame)
                self.draw_bpm(processed_frame)  # BPM 표시 추가
                
                # 처리된 프레임을 버퍼에 저장
                if not self.processed_frame_buffer.full():
                    self.processed_frame_buffer.put(processed_frame)
                    
            except queue.Empty:
                continue
                
    def update(self, frame, gesture_states, command):
        """메인 스레드에서 호출하여 프레임 데이터 전달"""
        if not self.frame_buffer.full():
            self.frame_buffer.put((frame.copy(), gesture_states, command))
            
    def get_processed_frame(self):
        """처리된 프레임 가져오기"""
        try:
            return self.processed_frame_buffer.get_nowait()
        except queue.Empty:
            return None
            
    def __del__(self):
        self.running = False
        if hasattr(self, 'gui_thread'):
            self.gui_thread.join()
        
    def calculate_fps(self):
        """FPS 계산"""
        import time
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = current_time
        return self.fps

    def draw_grid(self, frame):
        """화면을 4분할하는 그리드 그리기"""
        height, width = frame.shape[:2]
        mid_h, mid_w = height // 2, width // 2
        
        cv2.line(frame, (mid_w, 0), (mid_w, height), self.grid_color, 2)
        cv2.line(frame, (0, mid_h), (width, mid_h), self.grid_color, 2)
        return mid_h, mid_w

    def draw_section_texts(self, frame):
        """각 영역에 텍스트 표시"""
        height, width = frame.shape[:2]
        mid_h, mid_w = height // 2, width // 2
        font_scale = min(width, height) / 400

        centers = {
            AudioRegion.MELODY: (mid_w//2, mid_h//2),
            AudioRegion.VOCAL: (mid_w + mid_w//2, mid_h//2),
            AudioRegion.BASS: (mid_w//2, mid_h + mid_h//2),
            AudioRegion.DRUM: (mid_w + mid_w//2, mid_h + mid_h//2)
        }

        for region, center in centers.items():
            text = self.region_texts[region]
            text_size = cv2.getTextSize(text, self.font, font_scale, self.font_thickness)[0]
            text_x = center[0] - text_size[0]//2
            text_y = center[1] + text_size[1]//2
            cv2.putText(frame, text, (text_x, text_y),
                       self.font, font_scale, self.font_color, self.font_thickness)

    def draw_command_states(self, frame, command_states):
        """현재 명령어 상태를 화면에 표시"""
        font_scale = min(frame.shape[1], frame.shape[0]) / 1000
        
        regions = {
            "MELODY": [(10, 30), (255, 0, 0)],
            "VOCAL": [(frame.shape[1]-300, 30), (0, 255, 0)],
            "BASS": [(10, frame.shape[0]-90), (0, 0, 255)],
            "DRUM": [(frame.shape[1]-300, frame.shape[0]-90), (255, 255, 0)]
        }
        
        # 명령어 상태 업데이트
        if command_states.get(Command.NONE, 0) == 0:
            for region in regions.keys():
                if command_states.get(f"{region}_PLAY", 0) > 0:
                    self._states[region]["playback"] = "PLAYING"
                elif command_states.get(f"{region}_STOP", 0) > 0:
                    self._states[region]["playback"] = "STOPPED"
                
                # CONDUCTING 상태 업데이트
                self._states[region]["conducting"] = command_states.get(f"{region}_CONDUCT", 0) > 0
        
        # 상태 표시
        for region, (pos, color) in regions.items():
            state = self._states[region]
            
            # 첫 번째 줄: 재생 상태
            status = f"{region}: {state['playback']}"
            cv2.putText(frame, status, pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
            
            # 두 번째 줄: 볼륨
            volume_text = f"Volume: {state['volume']:.2f}"
            volume_pos = (pos[0], pos[1] + 25)
            cv2.putText(frame, volume_text, volume_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
            
            # 세 번째 줄: CONDUCTING 상태
            conducting_text = f"Conducting: {'ON' if state['conducting'] else 'OFF'}"
            conducting_pos = (pos[0], pos[1] + 50)
            cv2.putText(frame, conducting_text, conducting_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    def draw_hand_position(self, frame, gesture_states):
        """프레임에 왼손과 오른손의 위치를 사각형으로 표시"""
        if not gesture_states:
            return
        
        height, width = frame.shape[:2]
        rect_size = self.hand_marker_size * 2
        
        for hand in ["LEFT_", "RIGHT_"]:
            x = gesture_states.get(f"{hand}INDEX_TIP_X", None)
            y = gesture_states.get(f"{hand}INDEX_TIP_Y", None)
            
            if x is not None and y is not None:
                x, y = int(x * width), int(y * height)
                cv2.rectangle(frame,
                            (x - rect_size//2, y - rect_size//2),
                            (x + rect_size//2, y + rect_size//2),
                            self.hand_marker_color, 2)

    def draw_hand_landmarks(self, frame, multi_hand_landmarks, connections):
        """손의 랜드마크와 연결선 그리기"""
        if not multi_hand_landmarks:
            return
            
        height, width = frame.shape[:2]
        
        for hand_landmarks in multi_hand_landmarks:
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_point = hand_landmarks.landmark[start_idx]
                end_point = hand_landmarks.landmark[end_idx]
                
                start_x = int(start_point.x * width)
                start_y = int(start_point.y * height)
                end_x = int(end_point.x * width)
                end_y = int(end_point.y * height)
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y),
                        self.hand_marker_color, 2)

    def update_track_info(self, region, volume=None, conducting=None, soloed=None):
        """트랙 정보 업데이트"""
        if region in self._states:
            if volume is not None:
                self._states[region]["volume"] = volume
            if conducting is not None:
                self._states[region]["conducting"] = conducting
            if soloed is not None:
                self._states[region]["soloed"] = soloed

    def draw_audio_visualization(self, frame):
        """오디오 시각화 그리기"""
        height, width = frame.shape[:2]
        mid_h, mid_w = height // 2, width // 2
        
        regions = {
            "MELODY": (0, 0, mid_w, mid_h),
            "VOCAL": (mid_w, 0, width, mid_h),
            "BASS": (0, mid_h, mid_w, height),
            "DRUM": (mid_w, mid_h, width, height)
        }
        
        for region, (x1, y1, x2, y2) in regions.items():
            if region in self.visualizer.visualizer_data:
                data = self.visualizer.visualizer_data[region]
                viz_height = y2 - y1
                viz_width = x2 - x1
                
                # AudioController에서 전달받은 원본 RMS 값 사용
                original_rms = data.get("original_rms", 0)
                
                # STOPPED 상태이고 원본 RMS가 매우 작은 경우(무음)에만 회색으로 처리
                if self._states[region]["playback"] == "STOPPED" and original_rms < 0.001:
                    gray_overlay = np.full((viz_height, viz_width, 3), 128, dtype=np.uint8)
                    frame[y1:y2, x1:x2] = cv2.addWeighted(
                        frame[y1:y2, x1:x2], 0.3,
                        gray_overlay, 0.7,
                        0
                    )
                    continue
                
                # 현재 재생중인 오디오의 RMS로 시각화
                rms = np.sqrt(np.mean(np.array(data["waveform"])**2))
                normalized_rms = np.clip(rms / 0.2, 0, 1)
                
                # 1. 배경 그라데이션
                gradient = frame[y1:y2, x1:x2].copy()
                color1 = np.array(data["gradient"][0])
                color2 = np.array(data["gradient"][1])
                
                # PLAYING 상태일 때 색상 채도 증가
                if self._states[region]["playback"] == "PLAYING":
                    saturation_boost = 1.3  # 채도 증가 계수
                    color1 = np.clip(color1 * saturation_boost, 0, 255)
                    color2 = np.clip(color2 * saturation_boost, 0, 255)
                
                for i in range(viz_height):
                    ratio = i / viz_height
                    color = color1 * (1 - ratio) + color2 * ratio
                    cv2.line(gradient, (0, i), (viz_width, i), 
                            [int(c) for c in color], 1)
                
                # PLAYING 상태일 때 그라데이션 알파값 증가
                base_alpha = 0.2 if self._states[region]["playback"] == "PLAYING" else 0.1
                gradient_alpha = base_alpha + normalized_rms * 0.2
                
                cv2.addWeighted(gradient, gradient_alpha, 
                              frame[y1:y2, x1:x2], 1 - gradient_alpha, 
                              0, frame[y1:y2, x1:x2])
                
                # 2. 테두리 효과
                border_thickness = max(1, int(1 + normalized_rms * 2))
                border_color = data["color"]
                if self._states[region]["playback"] == "PLAYING":
                    border_color = tuple(np.clip(np.array(border_color) * 1.3, 0, 255))
                
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                            border_color, border_thickness)

    def draw_bpm(self, frame):
        """BPM 정보를 화면에 표시"""
        height, width = frame.shape[:2]
        bpm_text = f"BPM: {self.current_bpm:.1f}"
        font_scale = height / 720  # 화면 크기에 따라 폰트 크기 조절
        
        # 텍스트 크기 계산
        (text_width, text_height), _ = cv2.getTextSize(
            bpm_text, self.font, font_scale, self.font_thickness
        )
        
        # 텍스트 위치 계산 (중앙 상단)
        text_x = (width - text_width) // 2
        text_y = text_height + 20
        
        # BPM 텍스트 그리기
        cv2.putText(
            frame,
            bpm_text,
            (text_x, text_y),
            self.font,
            font_scale,
            self.bpm_color,
            self.font_thickness
        )

    def update_bpm(self, bpm):
        """현재 BPM 업데이트"""
        self.current_bpm = float(bpm)