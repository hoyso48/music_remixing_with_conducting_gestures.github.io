class Gesture:
    """제스처 상태 정의"""
    OPEN_PALM = "OPEN_PALM"
    CLOSED_FIST = "CLOSED_FIST"
    CONDUCTING = "CONDUCTING"
    SOLO = "SOLO"
    UNKNOWN = "UNKNOWN"
    
    @classmethod
    def get_all_gestures(cls):
        """모든 제스처 목록 반환"""
        return [cls.OPEN_PALM, cls.CLOSED_FIST, cls.CONDUCTING, cls.SOLO, cls.UNKNOWN]

class Command:
    """명령어 정의"""
    # 베이스 트랙 명령어
    BASS_PLAY = "BASS_PLAY"
    BASS_STOP = "BASS_STOP"
    
    # 보컬 트랙 명령어
    VOCAL_PLAY = "VOCAL_PLAY"
    VOCAL_STOP = "VOCAL_STOP"
    
    # 멜로디 트랙 명령어
    MELODY_PLAY = "MELODY_PLAY"
    MELODY_STOP = "MELODY_STOP"
    
    # 드럼 트랙 명령어
    DRUM_PLAY = "DRUM_PLAY"
    DRUM_STOP = "DRUM_STOP"
    
    # 기타
    NONE = "NONE"
    
    # CONDUCT 명령어 추가
    MELODY_CONDUCT = "MELODY_CONDUCT"
    VOCAL_CONDUCT = "VOCAL_CONDUCT"
    BASS_CONDUCT = "BASS_CONDUCT"
    DRUM_CONDUCT = "DRUM_CONDUCT"
    
    # SOLO 명령어 추가
    MELODY_SOLO = "MELODY_SOLO"
    VOCAL_SOLO = "VOCAL_SOLO"
    BASS_SOLO = "BASS_SOLO"
    DRUM_SOLO = "DRUM_SOLO"
    
    # MASTER 영역 명령어 추가
    MASTER_PLAY = "MASTER_PLAY"
    MASTER_STOP = "MASTER_STOP"
    MASTER_CONDUCT = "MASTER_CONDUCT"
    
    @classmethod
    def get_all_commands(cls):
        """모든 명령어 목록 반환"""
        return [
            cls.BASS_PLAY, cls.BASS_STOP,
            cls.VOCAL_PLAY, cls.VOCAL_STOP,
            cls.MELODY_PLAY, cls.MELODY_STOP,
            cls.DRUM_PLAY, cls.DRUM_STOP,
            cls.BASS_CONDUCT, cls.VOCAL_CONDUCT,
            cls.MELODY_CONDUCT, cls.DRUM_CONDUCT,
            cls.MELODY_SOLO, cls.VOCAL_SOLO,
            cls.BASS_SOLO, cls.DRUM_SOLO,
            cls.MASTER_PLAY, cls.MASTER_STOP,  # MASTER 명령어 추가
            cls.MASTER_CONDUCT,  # MASTER CONDUCT 추가
            cls.NONE
        ]

class AudioRegion:
    """오디오 영역 정의"""
    MELODY = "MELODY"
    VOCAL = "VOCAL"
    BASS = "BASS"
    DRUM = "DRUM"
    MASTER = "MASTER"  # 전체 음원 제어용
    
    @classmethod
    def get_all_regions(cls):
        """모든 오디오 영역 목록 반환"""
        return [cls.MELODY, cls.VOCAL, cls.BASS, cls.DRUM, cls.MASTER]

    @classmethod
    def get_file_path(cls, region):
        """각 영역에 해당하는 파일 경로 반환"""
        paths = {
            cls.BASS: "input2_60_gaudiolab_bass.mp3",
            cls.DRUM: "input2_60_gaudiolab_drum.mp3",
            cls.VOCAL: "input2_60_gaudiolab_vocal.mp3",
            cls.MELODY: "input2_60_gaudiolab_other.mp3"  # melody = other
        }
        return paths.get(region)  # MASTER는 None 반환