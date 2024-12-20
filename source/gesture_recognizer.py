import cv2
import mediapipe as mp
from collections import deque
from constants import Gesture, Command, AudioRegion
import numpy as np
import time

class GestureRecognizer:
    def __init__(self, history_size=512, ema_alpha=0.1):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.tempo_history_size = 512
        self.gesture_history = deque(maxlen=history_size)
        self.gesture_history_ema = deque(maxlen=history_size)  # EMA history 저장
        self.recognition_results_curr = None
        
        # EMA 관련 초기화
        self.ema_alpha = ema_alpha
        unknown_state = self._create_unknown_state()
        self.gesture_history_ema.append(unknown_state)  # 초기 EMA 상태

        
        # 초기 상태 설정 - 모든 영역이 STOPPED
        self._states = {
            "MELODY": {"playback": "STOPPED", "conducting": False, "volume": 0.0, "soloed": False},
            "VOCAL": {"playback": "STOPPED", "conducting": False, "volume": 0.0, "soloed": False},
            "BASS": {"playback": "STOPPED", "conducting": False, "volume": 0.0, "soloed": False},
            "DRUM": {"playback": "STOPPED", "conducting": False, "volume": 0.0, "soloed": False}
        }
        
    def process_frame(self, frame, use_ema=True):
        """프레임을 처리하고 현재 제스처 상태와 명령어를 반환"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        current_time = time.time() * 1000  # 현재 시간을 밀리초 단위로 변환
        self.recognition_results_curr = results
        current_gestures = {'timestamp': current_time}  # 시간 정보 추가
        
        # EMA를 위한 landmark 저장 구조 초기화 (처음 호출 시)
        if not hasattr(self, 'landmark_history_ema'):
            self.landmark_history_ema = {
                'Left': {'landmarks': None, 'weight': 0},
                'Right': {'landmarks': None, 'weight': 0}
            }
        
        if results.multi_hand_landmarks and results.multi_handedness:
            detected_hands = set()
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_side = handedness.classification[0].label
                detected_hands.add(hand_side)
                
                # 각 landmark에 대해 개별적으로 처리
                normalized_landmarks = []
                for i in range(21):  # MediaPipe는 21개의 landmark를 사용
                    landmark = hand_landmarks.landmark[i]
                    
                    # landmark가 유효한지 확인 (None이거나 비정상적인 값인 경우)
                    is_valid = (landmark is not None and 
                              not np.isnan(landmark.x) and 
                              not np.isnan(landmark.y) and 
                              not np.isnan(landmark.z))
                    
                    if is_valid:
                        # 유효한 landmark는 0~1로 정규화
                        x = np.clip(landmark.x, 0, 1)
                        y = np.clip(landmark.y, 0, 1)
                        z = np.clip(landmark.z, 0, 1)
                    else:
                        # 유효하지 않은 landmark는 이전 값 또는 0.5 사용
                        if (use_ema and 
                            self.landmark_history_ema[hand_side]['landmarks'] is not None):
                            prev_landmark = self.landmark_history_ema[hand_side]['landmarks'][i]
                            x, y, z = prev_landmark
                        else:
                            x = y = z = 0.5
                    
                    normalized_landmarks.append([x, y, z])
                
                current_landmarks = np.array(normalized_landmarks)
                

                # 정규화된 landmarks로 제스처 인식
                for i, landmark in enumerate(hand_landmarks.landmark):
                    landmark.x = normalized_landmarks[i][0]
                    landmark.y = normalized_landmarks[i][1]
                    landmark.z = normalized_landmarks[i][2]
                gesture_states = self.recognize_gesture(hand_landmarks, handedness)
                
                current_gestures.update(gesture_states)

            self.gesture_history.append(current_gestures)
            
            if use_ema:
                # EMA 업데이트 및 저장
                updated_ema = self._update_gesture_states_ema(current_gestures)
                self.gesture_history_ema.append(updated_ema)
                command_states = self.interpret_dommand()
                command_states['timestamp'] = current_time  # 시간 정보 추가
            else:
                # EMA 없이 직접 명령어 해석
                command_states = self.interpret_dommand()
                command_states['timestamp'] = current_time  # 시간 정보 추가
            
            return current_gestures, command_states
        
        unknown_state = self._create_unknown_state()
        unknown_state['timestamp'] = current_time  # 시간 정보 추가
        self.gesture_history.append(unknown_state)
        
        if use_ema:
            updated_ema = self._update_gesture_states_ema(unknown_state)
            self.gesture_history_ema.append(updated_ema)
            command_states = self.interpret_dommand()
            command_states['timestamp'] = current_time  # 시간 정보 추가
            return unknown_state, command_states
        else:
            command_states = self.interpret_dommand()
            command_states['timestamp'] = current_time  # 시간 정보 추가
            return unknown_state, command_states
    
    def _update_gesture_states_ema(self, current_gestures):
        """현재 gesture states를 사용하여 EMA 업데이트"""
        if not self.gesture_history_ema:
            return current_gestures.copy()
            
        prev_ema = self.gesture_history_ema[-1]
        updated_ema = {'timestamp': current_gestures['timestamp']}  # 시간은 필터링하지 않고 그대로 사용
        
        for key in current_gestures:
            if key == 'timestamp':  # timestamp는 EMA 처리하지 않음
                continue
            current_value = current_gestures[key]
            prev_value = prev_ema.get(key, current_value)
            updated_ema[key] = (self.ema_alpha * current_value + 
                              (1 - self.ema_alpha) * prev_value)
                              
        return updated_ema

    def is_conducting_gesture(self, hand_landmarks, handedness):
        """
        검지와 엄지가 붙어있는 상태의 confidence 반환
        """
        # 엄지와 검지 끝점
        finger_tips = [8, 12, 16, 20]  # 검지~새끼손가락 끝점
        finger_bases = [6, 10, 14, 18]  # 검지~새끼손가락 중간마디
        thumb_tip = 4  # 엄지 끝
        index_tip = 8  # 검지 끝
        
        # 두 점 사이의 거리 계산
        dx = hand_landmarks.landmark[thumb_tip].x - hand_landmarks.landmark[index_tip].x
        dy = hand_landmarks.landmark[thumb_tip].y - hand_landmarks.landmark[index_tip].y
        distance = (dx * dx + dy * dy) ** 0.5
        
        # 거리에 따른 confidence 계산 (거리가 max_distance 이하면 은 confidence)
        max_distance = 0.1
        confidence = max(0.0, 1.0 - (distance / max_distance))
        if dy < 0:
            confidence *= 0.3
        # 나머지 손가락 확인 (중지, 약지, 새끼손가락은 접혀있어야 함)
        for tip, base in zip(finger_tips[1:], finger_bases[1:]):  # 검지 제외하고 확인
            if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[base].y:  # 손가락이 접혀있으면
                confidence *= 0.3
        return confidence

    def is_open_palm(self, hand_landmarks, handedness):
        """
        손바닥이 펴져있는 정도의 confidence 반환
        """
        # 손가락 끝점들과 기준점
        finger_tips = [8, 12, 16, 20]  # 검지~새끼손가락 끝점
        finger_bases = [6, 10, 14, 18]  # 검지~새끼손가락 중간마디
        thumb_tip = 4
        thumb_base = 3

        # 왼손/오른손 구분
        is_right_hand = handedness.classification[0].label == "Right"

        # 손가락들이 펴져있는 정도 계산
        confidence = 1.0
        for tip, base in zip(finger_tips, finger_bases):
            if hand_landmarks.landmark[tip].y >= hand_landmarks.landmark[base].y:
                confidence *= 0.7
                
        # # 엄지 펴짐 정도 계산
        # if is_right_hand:
        #     if hand_landmarks.landmark[thumb_tip].x >= hand_landmarks.landmark[thumb_base].x:
        #         confidence *= 0.5
        # else:  # 왼손
        #     if hand_landmarks.landmark[thumb_tip].x <= hand_landmarks.landmark[thumb_base].x:
        #         confidence *= 0.5
            
        return confidence

    def is_closed_fist(self, hand_landmarks, handedness):
        """
        주먹을 쥐고 있는 정도의 confidence 환
        """
        # 손가락 끝점들과 기준점
        finger_tips = [8, 12, 16, 20]  # 검지~새끼손가락 끝점
        finger_bases = [6, 10, 14, 18]  # 검지~새끼손가락 중간마디
        
        # 손가락들이 접혀있는 정도 계산
        confidence = 1.0
        for tip, base in zip(finger_tips, finger_bases):
            if hand_landmarks.landmark[tip].y <= hand_landmarks.landmark[base].y:
                confidence *= 0.7
        
        return confidence

    def recognize_gesture(self, hand_landmarks, handedness):
        """제스처 인식 및 상태 반환"""
        hand_side = "LEFT_" if handedness.classification[0].label == "Left" else "RIGHT_"
        current_time = time.time() * 1000  # 현재 시간 (밀리초)
        
        # 검지 끝점(8번) 위치 저장
        index_tip = hand_landmarks.landmark[8]
        middle_base = hand_landmarks.landmark[9]
        
        gesture_states = {
            f"{hand_side}HAND_X": middle_base.x,
            f"{hand_side}HAND_Y": middle_base.y,
            f"{hand_side}INDEX_TIP_X": index_tip.x,
            f"{hand_side}INDEX_TIP_Y": index_tip.y,
        }
        
        # 각 제스처의 confidence 계산
        conducting_conf = self.is_conducting_gesture(hand_landmarks, handedness)
        open_palm_conf = self.is_open_palm(hand_landmarks, handedness)
        closed_fist_conf = self.is_closed_fist(hand_landmarks, handedness)
        solo_conf = self.is_solo_gesture(hand_landmarks, handedness)  # SOLO 제스처 추가
        
        # 모든 제스처에 대한 confidence 저장
        gesture_states.update({
            f"{hand_side}{Gesture.CONDUCTING}": conducting_conf,
            f"{hand_side}{Gesture.OPEN_PALM}": open_palm_conf,
            f"{hand_side}{Gesture.CLOSED_FIST}": closed_fist_conf,
            f"{hand_side}{Gesture.SOLO}": solo_conf,  # SOLO 상태 추가
            f"{hand_side}{Gesture.UNKNOWN}": 1.0 - max(conducting_conf, open_palm_conf, closed_fist_conf, solo_conf)
        })
        
        # SOLO 제스처 디버깅
        if solo_conf > 0.5:
            print(f"SOLO Gesture Detected - {hand_side} Hand")
            print(f"Position: ({index_tip.x:.2f}, {index_tip.y:.2f})")
        
        return gesture_states

    def get_region_from_position(self, x, y):
        """손의 위치로부터 해당하는 오디오 영역을 반환"""
        if x < 0.5:  # 왼쪽
            return AudioRegion.BASS if y > 0.5 else AudioRegion.MELODY
        else:  # 오른쪽
            return AudioRegion.DRUM if y > 0.5 else AudioRegion.VOCAL

    def calculate_conducting_area(self, conducting_hand_prefix, x, y):
        """지휘 동작 히스토리에서 면적 계산"""
        # 최근 10개의 기록만 사용
        recent_history = list(self.gesture_history)[-10:]
        if len(recent_history) < 2:
            return 0.0
            
        # conducting하는 손의 검지 끝점 좌표만 추출
        coords = [(g.get(f"{conducting_hand_prefix}INDEX_TIP_X", x),
                  g.get(f"{conducting_hand_prefix}INDEX_TIP_Y", y))
                 for g in recent_history]
        
        if len(coords) < 2:
            return 0.0
            
        # 히스토리에서 x, y의 최대/최소값으로 면적 계산
        x_coords, y_coords = zip(*coords)
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        return width * height

    def calculate_tempo(self, conducting_hand_prefix):
        """
        지휘 동작의 tempo를 계산
        Returns:
            float: tempo period (초 단위), 지휘 동작이 없으면 None
        """
        recent_history = list(self.gesture_history_ema)[-self.tempo_history_size:]
        if len(recent_history) < 20:
            print('tempo_history_size is not enough')
            return None
        
        # 최근 10개의 기록만 확인
        last_10_records = recent_history[-10:]
        conducting_active = [g.get(f"{conducting_hand_prefix}{Gesture.CONDUCTING}", 0) > 0.5 
                            for g in last_10_records]
        
        # 최근 10개 모두가 지휘 동작이어야 함
        if sum(conducting_active) < 10:  # 10개 모두 True여야 함
            return None
        
        # conducting하는 손의 x, y 좌표 추출
        y_coords = [g.get(f"{conducting_hand_prefix}INDEX_TIP_X", 0.5) for g in recent_history]
        
        # peak 찾기 (최근부터 과거로)
        peaks = []
        for i in range(len(y_coords)-2, 1, -1):
            if y_coords[i] > y_coords[i+1] and y_coords[i] >= y_coords[i-1]:
                peaks.append(i)
                if len(peaks) == 2:
                    break

        if len(peaks) < 2:
            return None
        
        # 두 peak 간의 거리 계산
        peak_distance = peaks[0] - peaks[1]
        
        # peak_distance를 BPM으로 변환 (30fps 가정)
        tempo_period = 2 * peak_distance / (27.5*2)  # 초 단위
        bpm = 60.0 / tempo_period

        # 합리적인 tempo 범위로 BPM 제한 (20-180 BPM)
        if not (20 <= bpm <= 200):
            return None

        print(f"Peak indices: {peaks}, Distance: {peak_distance}, BPM: {bpm:.1f}")
        return bpm

    def interpret_dommand(self):
        """EMA history를 사용한 command 해석"""
        command_states = {cmd: 0.0 for cmd in Command.get_all_commands()}
        
        if len(self.gesture_history) < 2:
            command_states[Command.NONE] = 1.0
            return command_states
        
        current_ema = self.gesture_history[-1]
        prev_ema = self.gesture_history[-2]
        
        # SOLO 처리를 최우선으로 변경
        # 1. SOLO 처리
        solo_detected = False
        for hand_prefix in ["LEFT_", "RIGHT_"]:
            current_solo = current_ema.get(f"{hand_prefix}{Gesture.SOLO}", 0)
            prev_solo = prev_ema.get(f"{hand_prefix}{Gesture.SOLO}", 0)
            
            # SOLO 제스처가 있을 때
            if current_solo > 0.5:
                solo_detected = True
                x = current_ema.get(f"{hand_prefix}INDEX_TIP_X", 0.5)
                y = current_ema.get(f"{hand_prefix}INDEX_TIP_Y", 0.5)
                region = self.get_region_from_position(x, y)
                command_states[f"{region}_SOLO"] = 1.0
                print(f"SOLO ON: {region}")  # 디버깅
            
            # SOLO 제스처가 끝났을 때 (1 -> 0 전환)
            elif prev_solo > 0.5 and current_solo <= 0.5:
                print(f"SOLO OFF transition detected")  # 디버깅
                # 모든 SOLO 명령을 명시적으로 0으로 설정
                for region in AudioRegion.get_all_regions():
                    if region != AudioRegion.MASTER:
                        command_states[f"{region}_SOLO"] = 0.0
                return command_states  # SOLO OFF 시 다른 명령 무시하고 즉시 반환
        
        # 2. STOP/PLAYING 처리 (두 번째 우선순위)
        for hand_prefix in ["LEFT_", "RIGHT_"]:
            x = current_ema.get(f"{hand_prefix}HAND_X", 0.5)
            y = current_ema.get(f"{hand_prefix}HAND_Y", 0.5)
            
            region_commands = {
                (False, False): (Command.MELODY_PLAY, Command.MELODY_STOP),
                (True, False): (Command.VOCAL_PLAY, Command.VOCAL_STOP),
                (False, True): (Command.BASS_PLAY, Command.BASS_STOP),
                (True, True): (Command.DRUM_PLAY, Command.DRUM_STOP)
            }
            
            play_command, stop_command = region_commands.get((x > 0.5, y > 0.5), (None, None))
            
            if play_command and stop_command:
                # PLAY 처리
                curr_open_palm = current_ema.get(f"{hand_prefix}{Gesture.OPEN_PALM}", 0) > 0.5
                prev_closed_fist = prev_ema.get(f"{hand_prefix}{Gesture.CLOSED_FIST}", 0) > 0.5
                if curr_open_palm and prev_closed_fist:
                    command_states[play_command] = 1.0
                
                # STOP 처리
                curr_closed_fist = current_ema.get(f"{hand_prefix}{Gesture.CLOSED_FIST}", 0) > 0.5
                prev_open_palm = prev_ema.get(f"{hand_prefix}{Gesture.OPEN_PALM}", 0) > 0.5
                if curr_closed_fist and prev_open_palm:
                    command_states[stop_command] = 1.0
        
        # 3. CONDUCTING 처리 (마지막 우선순위)
        left_conducting = current_ema.get(f"LEFT_{Gesture.CONDUCTING}", 0) > 0.5
        right_conducting = current_ema.get(f"RIGHT_{Gesture.CONDUCTING}", 0) > 0.5
        
        for hand_prefix in ["LEFT_", "RIGHT_"]:
            is_conducting = current_ema.get(f"{hand_prefix}{Gesture.CONDUCTING}", 0) > 0.5
            if is_conducting:
                x = current_ema.get(f"{hand_prefix}INDEX_TIP_X", 0.5)
                y = current_ema.get(f"{hand_prefix}INDEX_TIP_Y", 0.5)
                
                region = self.get_region_from_position(x, y)
                area = self.calculate_conducting_area(hand_prefix, x, y)
                min_area = 0.0
                max_area = 0.25
                scale = 4.0
                volume = (area - min_area) / (max_area - min_area) * scale
                
                command_states[f"{region}_CONDUCT"] = 1.0
                command_states[f"{region}_CONDUCT_X"] = x
                command_states[f"{region}_CONDUCT_Y"] = y
                command_states[f"{region}_CONDUCT_VOLUME"] = volume
        
        # Tempo 계산 (RIGHT 우선, 없으면 LEFT)
        if right_conducting:
            tempo_period = self.calculate_tempo("RIGHT_")
            if tempo_period is not None:
                command_states["TEMPO"] = tempo_period
        elif left_conducting:
            tempo_period = self.calculate_tempo("LEFT_")
            if tempo_period is not None:
                command_states["TEMPO"] = tempo_period
        
        if all(v == 0.0 for v in command_states.values()):
            command_states[Command.NONE] = 1.0
        
        return command_states

    def _create_unknown_state(self):
        unknown_state = {}
        for hand_prefix in ["LEFT_", "RIGHT_"]:
            unknown_state.update({
                f"{hand_prefix}{gesture}": 0.0 for gesture in Gesture.get_all_gestures()
            })
            unknown_state[f"{hand_prefix}{Gesture.UNKNOWN}"] = 1.0
            unknown_state.update({
                f"{hand_prefix}HAND_X": 0.5,
                f"{hand_prefix}HAND_Y": 0.5
            })
            unknown_state.update({
                f"{hand_prefix}INDEX_TIP_X": 0.5,
                f"{hand_prefix}INDEX_TIP_Y": 0.5
            })
        return unknown_state

    @property
    def hand_connections(self):
        return self.mp_hands.HAND_CONNECTIONS

    def is_solo_gesture(self, hand_landmarks, handedness):
        """
        검지와 엄지만 펴고 나머지는 접힌 상태의 confidence 반환
        """
        # 손가락 끝점들과 기준점
        finger_tips = [8, 12, 16, 20]  # 검지~새끼손가락 끝점
        finger_bases = [6, 10, 14, 18]  # 검지~새끼손가락 중간마디
        thumb_tip = 4
        thumb_base = 3

        # 왼손/오른손 구분
        is_right_hand = handedness.classification[0].label == "Right"
        
        confidence = 1.0
        
        # 검지 확인 (펴져 있어야 함)
        if hand_landmarks.landmark[8].y >= hand_landmarks.landmark[6].y:  # 검지가 접혀있으면
            confidence *= 0.3
        
        # 엄지 확인 (펴져 있어야 함)
        if is_right_hand:
            if hand_landmarks.landmark[thumb_tip].x >= hand_landmarks.landmark[thumb_base].x:  # 엄지가 접혀있으면
                confidence *= 0.3
        else:  # 왼손
            if hand_landmarks.landmark[thumb_tip].x <= hand_landmarks.landmark[thumb_base].x:  # 엄지 접혀있으면
                confidence *= 0.3
            
        # 나머지 손가락 확인 (중지, 약지, 새끼손가락은 접혀있어야 함)
        for tip, base in zip(finger_tips[1:], finger_bases[1:]):  # 검지 제외하고 확인
            if hand_landmarks.landmark[tip].y <= hand_landmarks.landmark[base].y:  # 손가락이 펴져있으면
                confidence *= 0.3
        
        return confidence

    def reset_ema(self):
        """EMA 관련 변수들을 초기화"""
        self.landmark_history_ema = {
            'Left': {'landmarks': None, 'weight': 0},
            'Right': {'landmarks': None, 'weight': 0}
        }
        self.gesture_history = []
        self.gesture_history_ema = []