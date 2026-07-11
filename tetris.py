# -*- coding: utf-8 -*-
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass, field as dataclass_field


vertex = np.array([
( 0.0, 0.0, 0.0 ), # A
( 1.0, 0.0, 0.0 ), # B
( 1.0, 1.0, 0.0 ), # C
( 0.0, 1.0, 0.0 ), # D
( 0.0, 0.0, 1.0 ), # E
( 1.0, 0.0, 1.0 ), # F
( 1.0, 1.0, 1.0 ), # G
( 0.0, 1.0, 1.0 ) # H
])

edge = np.array([
[0, 1], [1, 2], [2, 3], [3, 0], # 髯溷叙蝮ｩ隰ｫ繝ｻ・ｸ・ｺ繝ｻ・ｮ鬮ｴ雜｣・ｽ・ｺ
[4, 5], [5, 6], [6, 7], [7, 4], # 髣包ｽｳ闔ｨ竏ｵ蟲ｶ驍ｵ・ｺ繝ｻ・ｮ鬮ｴ雜｣・ｽ・ｺ
[0, 4], [1, 5], [2, 6], [3, 7]  # 髯句ｹ｢・ｽ・ｴ鬯ｮ・ｱ繝ｻ・｢驍ｵ・ｺ繝ｻ・ｮ鬮ｴ雜｣・ｽ・ｺ
])

# 髫ｴ蠑ｱ・翫・・ｨ闔・･陞ｻ骰具ｽｹ・ｧ驗呻ｽｫ遶企ｦｴ蝮弱・・ｭ髯橸ｽｳ郢晢ｽｻ
face = np.array([
[3, 2, 1, 0], # D-C-B-A 驛｢・ｧ陜｣・､繝ｻ・ｵ髣雁ｾ後・鬯ｮ・ｱ繝ｻ・｢
[2, 1, 5, 6], # C-B-F-G 驛｢・ｧ陜｣・､繝ｻ・ｵ髣雁ｾ後・鬯ｮ・ｱ繝ｻ・｢
[6, 5, 4, 7], # G-F-E-H 驛｢・ｧ陜｣・､繝ｻ・ｵ髣雁ｾ後・鬯ｮ・ｱ繝ｻ・｢
[3, 7, 4, 0], # D-H-E-A 驛｢・ｧ陜｣・､繝ｻ・ｵ髣雁ｾ後・鬯ｮ・ｱ繝ｻ・｢
[0, 4, 5, 1], # A-E-F-B 驛｢・ｧ陜｣・､繝ｻ・ｵ髣雁ｾ後・鬯ｮ・ｱ繝ｻ・｢
[7, 3, 2, 6]  # H-D-C-G 驛｢・ｧ陜｣・､繝ｻ・ｵ髣雁ｾ後・鬯ｮ・ｱ繝ｻ・｢
])

normals = np.array([
[ 0.0, 0.0,-1.0], # D-C-B-A 驛｢・ｧ陜｣・､繝ｻ・ｵ髣雁ｾ後・鬯ｮ・ｱ繝ｻ・｢驍ｵ・ｺ繝ｻ・ｮ髮主｢薙・繝ｻ・ｷ郢晢ｽｻ
[ 0.0, 1.0, 0.0], # C-B-F-G 驛｢・ｧ陜｣・､繝ｻ・ｵ髣雁ｾ後・鬯ｮ・ｱ繝ｻ・｢驍ｵ・ｺ繝ｻ・ｮ髮主｢薙・繝ｻ・ｷ郢晢ｽｻ
[ 0.0, 0.0, 1.0], # G-F-E-H 驛｢・ｧ陜｣・､繝ｻ・ｵ髣雁ｾ後・鬯ｮ・ｱ繝ｻ・｢驍ｵ・ｺ繝ｻ・ｮ髮主｢薙・繝ｻ・ｷ郢晢ｽｻ
[ 0.0,-1.0, 0.0], # D-H-E-A 驛｢・ｧ陜｣・､繝ｻ・ｵ髣雁ｾ後・鬯ｮ・ｱ繝ｻ・｢驍ｵ・ｺ繝ｻ・ｮ髮主｢薙・繝ｻ・ｷ郢晢ｽｻ
[-1.0, 0.0, 0.0], # A-E-F-B 驛｢・ｧ陜｣・､繝ｻ・ｵ髣雁ｾ後・鬯ｮ・ｱ繝ｻ・｢驍ｵ・ｺ繝ｻ・ｮ髮主｢薙・繝ｻ・ｷ郢晢ｽｻ
[ 1.0, 0.0, 0.0]  # H-D-C-G 驛｢・ｧ陜｣・､繝ｻ・ｵ髣雁ｾ後・鬯ｮ・ｱ繝ｻ・｢驍ｵ・ｺ繝ｻ・ｮ髮主｢薙・繝ｻ・ｷ郢晢ｽｻ
])

minos = {
    "I": np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]]),
    "O": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
    "T": np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]]),
    "S": np.array([[0, 0, 0], [1, 0, 0], [-1, 1, 0], [0, 1, 0]]),
    "Z": np.array([[-1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]]),
    "J": np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0], [-1, 1, 0]]),
    "L": np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]])
}

kick_tests = [
    [0, 0, 0],
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [2, 0, 0],
    [-2, 0, 0],
]

# 雎ｼ・ｶ繝ｻ・ｲ驍ｵ・ｺ繝ｻ・ｮ髯橸ｽｳ陞溘ｑ・ｽ・ｾ繝ｻ・ｩ
COLORS = {
    "I": (0.0, 1.0, 1.0),  # 驛｢・ｧ繝ｻ・ｷ驛｢・ｧ繝ｻ・｢驛｢譎｢・ｽ・ｳ
    "O": (1.0, 1.0, 0.0),  # 鬲・ｺ倥・
    "T": (1.0, 0.0, 1.0),  # 驛｢譎・ｽｧ・ｭ邵ｺ讓抵ｽｹ譎｢・ｽ・ｳ驛｢・ｧ繝ｻ・ｿ
    "S": (0.0, 1.0, 0.0),  # 鬩搾ｽｱ郢晢ｽｻ
    "Z": (1.0, 0.0, 0.0),  # 髫俶誓・ｽ・､
    "J": (0.0, 0.0, 1.0),  # 鬯ｮ・ｱ郢晢ｽｻ
    "L": (1.0, 0.5, 0.0)   # 驛｢・ｧ繝ｻ・ｪ驛｢譎｢・ｽ・ｬ驛｢譎｢・ｽ・ｳ驛｢・ｧ繝ｻ・ｸ
}

# 驛｢譎・ｽｼ譁絶襖驛｢譎｢・ｽ・ｼ驛｢譎｢・ｽ・ｫ驛｢譎擾ｽｳ・ｨ遶頑･｢蟆・ｭ取得・ｽ・ｭ陋滂ｽ･隨倥・・ｹ・ｧ闕ｵ譏ｶ陞ｺ驛｢・ｧ遶丞｣ｹ繝ｻ雎ｼ・ｶ繝ｻ・ｲID
COLOR_IDS = {"I": 1, "O": 2, "T": 3, "S": 4, "Z": 5, "J": 6, "L": 7}
# 髫ｰ・ｰ陷諤懈・髫ｴ蠑ｱ・・ｫ顔噪D驍ｵ・ｺ闕ｵ譎｢・ｽ逕ｻ・ｿ・ｶ繝ｻ・ｲ驛｢・ｧ髮区ｧｫ蠕宣Δ・ｧ鬮ｮ竏壹・驍ｵ・ｺ陷ｷ・ｶ隨ｳ繝ｻ・ｹ・ｧ遶丞｣ｹ繝ｻ鬮ｴ蜿厄ｽｨ髮・ｽｶ繝ｻ
ID_TO_COLOR = {1: COLORS["I"], 2: COLORS["O"], 3: COLORS["T"], 4: COLORS["S"], 5: COLORS["Z"], 6: COLORS["J"], 7: COLORS["L"]}

@dataclass
class GameState:
    current_mino_type: str | None = None
    next_mino_type: str | None = None
    next_mino_queue: list = dataclass_field(default_factory=list)
    hold_mino_type: str | None = None
    can_hold: bool = True

    mino_pos: np.ndarray | None = None
    mino: np.ndarray | None = None
    field: np.ndarray | None = None

    score: int = 0
    total_lines_cleared: int = 0
    last_total_lines_cleared: int = 0

    game_over: bool = False
    drop_switch: bool = True

    last_drop_time: float = 0.0
    lock_timer: float | None = None
    lock_reset_counter: int = 0
    pause_started_time: float | None = None

    mino_bag: list = dataclass_field(default_factory=list)

state = GameState()

# ==========================
# 郢晁ｼ斐≦郢晢ｽｼ郢晢ｽｫ郢晁歓・ｨ・ｭ陞ｳ繝ｻ
# ==========================
FIELD_WIDTH = 10
FIELD_HEIGHT = 20
SPAWN_POS = np.array([4.0, 19.0, 0.0])

# ==========================
# 郢ｧ・ｲ郢晢ｽｼ郢晢｣ｰ郢晢ｽｫ郢晢ｽｼ郢晢ｽｫ髫ｪ・ｭ陞ｳ繝ｻ
# ==========================
NEXT_QUEUE_SIZE = 5
LOCK_DELAY = 0.5
MAX_LOCK_RESETS = 15
LINES_PER_LEVEL = 10
MIN_FALL_INTERVAL = 0.1
INITIAL_FALL_INTERVAL = 1.0
FALL_INTERVAL_DECREASE_PER_LEVEL = 0.1

# ==========================
# 郢ｧ・ｹ郢ｧ・ｳ郢ｧ・｢髫ｪ・ｭ陞ｳ繝ｻ
# ==========================
SCORE_TABLE = {
    1: 100,
    2: 300,
    3: 500,
    4: 800,
}

# ==========================
# 郢ｧ・ｦ郢ｧ・｣郢晢ｽｳ郢晏ｳｨ縺磯坎・ｭ陞ｳ繝ｻ
# ==========================
WINDOW_WIDTH = 512
WINDOW_HEIGHT = 768
WINDOW_TITLE = "3D Tetris"

# ==========================
# 郢ｧ・ｫ郢晢ｽ｡郢晢ｽｩ郢晢ｽｻ郢晢ｽｩ郢ｧ・､郢晞メ・ｨ・ｭ陞ｳ繝ｻ
# ==========================
CAMERA_EYE = (15.0, 10.0, 30.0)
CAMERA_CENTER = (5.0, 10.0, 0.0)
CAMERA_UP = (0.0, 1.0, 0.0)

LIGHT_POSITION = [5.0, 10.0, 20.0, 1.0]

# ==========================
# 髢ｭ譴ｧ蜍ｹ髫ｪ・ｭ陞ｳ繝ｻ
# ==========================
FLOOR_SIZE = 12
BACKGROUND_SIZE = 22
FLOOR_POS = (5.0, 0.0, -1.0)
BACKGROUND_POS = (5.0, 10.0, -1.0)

# ==========================
# UI鬩溷調・ｽ・ｮ
# ==========================
NEXT_PREVIEW_POS = (13.0, 18.0, 0.0)
NEXT_PREVIEW_INTERVAL_Y = 4.0

HOLD_PREVIEW_POS = (-4.0, 15.0, 0.0)

SCORE_POS = (13.0, -1.0, 0.0)
DIGIT_SPACING = 1.2

PREVIEW_FRAME_LEFT = -2.0
PREVIEW_FRAME_RIGHT = 4.0
PREVIEW_FRAME_BOTTOM = -1.0
PREVIEW_FRAME_TOP = 3.0
PREVIEW_FRAME_Z = 1.01

# ==========================
# 隰蜀怜愛郢ｧ・ｹ郢ｧ・ｿ郢ｧ・､郢晢ｽｫ
# ==========================
BLOCK_SPECULAR = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
BLOCK_SHININESS = 128.0

GHOST_ALPHA = 0.3
GHOST_SHININESS = 32.0

BLOCK_EDGE_COLOR = (0.0, 0.0, 0.0)
BLOCK_EDGE_WIDTH = 2.0

FIELD_GRID_COLOR = (0.7, 0.7, 0.7)
FIELD_GRID_WIDTH = 2.0
FIELD_OUTLINE_WIDTH = 4.0
FIELD_FRAME_Z = 1.01

# ==========================
# 郢ｧ・｢郢ｧ・ｻ郢昴・繝ｨ郢昜ｻ｣縺・
# ==========================
BASE_DIR = Path(__file__).resolve().parent
DIGIT_DIR = BASE_DIR / "digits"

# ==========================
# 郢ｧ・ｭ郢晢ｽｼ髫ｪ・ｭ陞ｳ繝ｻ
# ==========================
KEY_QUIT = glfw.KEY_U
KEY_RESTART = glfw.KEY_R
KEY_PAUSE = glfw.KEY_ESCAPE

KEY_MOVE_LEFT = glfw.KEY_LEFT
KEY_MOVE_RIGHT = glfw.KEY_RIGHT
KEY_SOFT_DROP = glfw.KEY_DOWN
KEY_HARD_DROP = glfw.KEY_SPACE

KEY_ROTATE_RIGHT = (glfw.KEY_UP, glfw.KEY_E)
KEY_ROTATE_LEFT = (glfw.KEY_Q, glfw.KEY_LEFT_CONTROL)
KEY_HOLD = (glfw.KEY_C, glfw.KEY_LEFT_SHIFT)

def initTextureFromFile(filename):

    with Image.open(filename) as img:
        # glTexImage2D 驍ｵ・ｺ繝ｻ・ｫ髮九ｑ・ｽ・｡驍ｵ・ｺ陷ｷ譎｢・ｽ・ｽ繝ｻ・｢髯溷床・ｸ螂・ｽｽ繝ｻRGB 驍ｵ・ｺ繝ｻ・ｫ鬩搾ｽｨ繝ｻ・ｱ髣包ｽｳ・つ驍ｵ・ｺ陷ｷ・ｶ繝ｻ迢暦ｽｸ・ｲ郢晢ｽｻ
        image = img.convert("RGB")
        # PIL 驍ｵ・ｺ繝ｻ・ｨ OpenGL 驍ｵ・ｺ繝ｻ・ｧ驍ｵ・ｺ繝ｻ・ｯ鬨ｾ蛹・ｽｽ・ｻ髯ｷ蜑・ｽｸ鄙ｫ繝ｻ髯ｷ・ｴ雋・ｽｽ邵ｺ蟶ｷ・ｸ・ｺ陟包ｽ｡繝ｻ・ｸ髮榊・・ｽ・ｸ驕擾ｽｩ・つ郢晢ｽｻ遶企・・ｸ・ｺ繝ｻ・ｮ驍ｵ・ｺ繝ｻ・ｧ髯ｷ・ｿ陝雜｣・ｽ・ｻ繝ｻ・｢驍ｵ・ｺ陷ｷ・ｶ繝ｻ迢暦ｽｸ・ｲ郢晢ｽｻ
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        width, height = image.size
        data = image.tobytes()

    texture_id = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, texture_id)

    # 驛｢譏ｴ繝ｻ邵ｺ驢搾ｽｹ・ｧ繝ｻ・ｹ驛｢譏ｶ繝ｻ・取坩ﾂ蛹・ｽｽ・ｻ髯ｷ蜑・ｽｸ鄙ｫ繝ｻ驛｢譎√・邵ｺ繝ｻ・ｹ譏懶ｽｺ・･髢ｻ・ｰ髣厄ｽｴ鬮ｦ・ｪ遶企ｦｴ蝨ｦ繝ｻ・ｰ驛｢・ｧ遶擾ｽｬ繝ｻ・ｾ繝ｻ・ｼ驍ｵ・ｺ繝ｻ・ｾ驛｢・ｧ陟募ｨｯﾂ・ｻ驍ｵ・ｺ郢晢ｽｻ繝ｻ迢暦ｽｸ・ｲ郢晢ｽｻ
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        width,
        height,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        data,
    )

    # 驛｢譏ｴ繝ｻ邵ｺ驢搾ｽｹ・ｧ繝ｻ・ｹ驛｢譏ｶ繝ｻ・取・・ｸ・ｺ繝ｻ・ｮ鬮ｯ・ｬ隲橸ｽｺ闖ｫ・｣髫ｴ繝ｻ・ｽ・ｹ髮取・・ｼ雋ｻ・ｽ螳夲ｽｬ謔ｶ繝ｻ繝ｻ・ｮ陞｢・ｹ隨倥・・ｹ・ｧ闕ｵ謨鳴郢晢ｽｻ
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_2D, 0)
    
    return texture_id

# 驛｢・ｧ繝ｻ・ｲ驛｢譎｢・ｽ・ｼ驛｢譎｢・｣・ｰ髴托ｽ･繝ｻ・ｶ髫ｲ・ｷ闕ｵ譎｢・ｽ螳壼ｴ戊ｭ弱・・・刹・ｹ陷ｴ繝ｻ・ｽ・ｼ陋ｹ・ｻ・取㏍・ｹ・ｧ繝ｻ・ｻ驛｢譏ｴ繝ｻ郢晢ｽｨ郢晢ｽｻ陝ｲ・ｨ隨倥・・ｹ・ｧ驕擾ｽｩ隴幢ｽｪ髫ｰ・ｨ繝ｻ・ｰ
def reset_game():
    
    state.score = 0
    state.total_lines_cleared = 0
    state.last_total_lines_cleared = 0
    state.lock_reset_counter = 0
    state.lock_timer = None
    state.pause_started_time = None

    state.mino_bag.clear()  # 驛｢譎・ｽｺ蛟･ﾎ驍ｵ・ｺ繝ｻ・ｮ驛｢譎√・邵ｺ蝣､・ｹ・ｧ陋幢ｽｵ邵ｺ驢搾ｽｹ譎｢・ｽ・ｪ驛｢・ｧ繝ｻ・｢
    state.current_mino_type = get_next_mino_type()
    state.next_mino_queue.clear()  # 驛｢譎樔ｺらｸｺ驢搾ｽｹ・ｧ繝ｻ・ｹ驛｢譎冗樟邵ｺ蜀暦ｽｹ譎｢・ｽ・･驛｢譎｢・ｽ・ｼ驛｢・ｧ髮区ｧｭ繝ｻ髫ｴ蟶ｶ・ｺ・ｷ陜滂ｽｧ
    refill_next_queue()
    state.next_mino_type = state.next_mino_queue[0]  # 驛｢譎樔ｺらｸｺ驢搾ｽｹ・ｧ繝ｻ・ｹ驛｢譎冗樟邵ｺ蜀暦ｽｹ譎｢・ｽ・･驛｢譎｢・ｽ・ｼ驍ｵ・ｺ繝ｻ・ｮ髯ｷ閧ｲ・｣・ｯ繝ｻ・ｰ繝ｻ・ｭ驛｢・ｧ陷ｻ闌ｨ・ｽ・ｬ繝ｻ・｡驍ｵ・ｺ繝ｻ・ｮ驛｢譎・ｽｺ蛟･ﾎ驍ｵ・ｺ繝ｻ・ｨ驍ｵ・ｺ陷会ｽｱ遯ｶ・ｻ鬮ｫ・ｪ繝ｻ・ｭ髯橸ｽｳ郢晢ｽｻ
    state.hold_mino_type = None
    state.can_hold = True
    state.mino_pos = SPAWN_POS.copy()
    state.mino = np.copy(minos[state.current_mino_type])
    
    state.drop_switch = True
    state.last_drop_time = glfw.get_time()
    state.field = np.zeros((FIELD_HEIGHT, FIELD_WIDTH), dtype=int) 
    state.game_over = False
    print("Game Reset!")
# ==========================


def get_next_mino_type():
    
    if not state.mino_bag:
        state.mino_bag.extend(list(minos.keys()))
        np.random.shuffle(state.mino_bag)
    return state.mino_bag.pop()

def refill_next_queue():
    while len(state.next_mino_queue) < NEXT_QUEUE_SIZE:
        state.next_mino_queue.append(get_next_mino_type())
        
# 驛｢譎・ｽｺ蛟･ﾎ驛｢・ｧ陋幢ｽｵ郢晢ｽｵ驛｢・ｧ繝ｻ・｣驛｢譎｢・ｽ・ｼ驛｢譎｢・ｽ・ｫ驛｢譎擾ｽｳ・ｨ遶頑･｢鞫弱・・ｺ髯橸ｽｳ陞｢・ｹ繝ｻ・ｰ驍ｵ・ｲ遶擾ｽｵ騾ｵ・ｰ驍ｵ・ｺ陷会ｽｱ繝ｻ讓抵ｽｹ譎・ｽｺ蛟･ﾎ驛｢・ｧ陜｣・､陷・ｽｽ髫ｰ迹壹・隨倥・・ｹ・ｧ郢晢ｽｻ
def lock_and_spawn_mino():

    if state.game_over:
        return  # 驛｢・ｧ繝ｻ・ｲ驛｢譎｢・ｽ・ｼ驛｢譎｢・｣・ｰ驛｢・ｧ繝ｻ・ｪ驛｢譎｢・ｽ・ｼ驛｢譎√・郢晢ｽｻ髫ｴ蠑ｱ・・ｹ晢ｽｻ髣厄ｽｴ髴郁ｲｻ・ｽ繧会ｽｸ・ｺ陷会ｽｱ遶企・・ｸ・ｺ郢晢ｽｻ

    # 髯懈圜・ｽ・ｺ髯橸ｽｳ陞｢・ｼ郢晢ｽｻ鬨ｾ繝ｻ繝ｻ
    color_id = COLOR_IDS[state.current_mino_type]
    # 驛｢譎・ｽｺ蛟･ﾎ驍ｵ・ｺ繝ｻ・ｮ4驍ｵ・ｺ繝ｻ・､驍ｵ・ｺ繝ｻ・ｮ驛｢譎・§・取ｺｽ・ｹ譏ｴ繝ｻ邵ｺ驢搾ｽｹ・ｧ陋幢ｽｵ郢晢ｽｵ驛｢・ｧ繝ｻ・｣驛｢譎｢・ｽ・ｼ驛｢譎｢・ｽ・ｫ驛｢譎擾ｽｳ・ｨ遶頑･｢鞫弱・・ｺ髯橸ｽｳ郢晢ｽｻ
    for i in range(4):
        x = int(state.mino_pos[0] + state.mino[i][0])
        y = int(state.mino_pos[1] + state.mino[i][1])
        if 0 <= x < FIELD_WIDTH and 0 <= y < FIELD_HEIGHT and state.field[y][x] == 0:
            state.field[y][x] = color_id

    # 鬮ｯ・ｦ陟募ｾ後・驛｢・ｧ繝ｻ・ｯ驛｢譎｢・ｽ・ｪ驛｢・ｧ繝ｻ・｢驛｢・ｧ陋幢ｽｵ郢晢ｽ｡驛｢・ｧ繝ｻ・ｧ驛｢譏ｴ繝ｻ邵ｺ繝ｻ
    lines_cleared = check_line_clear()
    state.score += SCORE_TABLE.get(lines_cleared, 0)
    
    # 驛｢譎冗樟郢晢ｽｻ驛｢・ｧ繝ｻ・ｿ驛｢譎｢・ｽ・ｫ驛｢譎｢・ｽ・ｩ驛｢・ｧ繝ｻ・､驛｢譎｢・ｽ・ｳ髫ｰ・ｨ繝ｻ・ｰ驛｢・ｧ陷ｻ莠･・ｳ・ｩ髫ｴ繝ｻ・ｽ・ｰ
    state.total_lines_cleared += lines_cleared
    
    # 驛｢譎｢・ｽ・ｬ驛｢譎冗函・取刮・ｹ・ｧ繝ｻ・｢驛｢譏ｴ繝ｻ郢晢ｽｻ驍ｵ・ｺ繝ｻ・ｮ髯具ｽｻ繝ｻ・､髯橸ｽｳ陞滂ｽｲ繝ｻ・ｼ郢晢ｽｻ0驛｢譎｢・ｽ・ｩ驛｢・ｧ繝ｻ・､驛｢譎｢・ｽ・ｳ驍ｵ・ｺ隴∫ｵｶ繝ｻ驍ｵ・ｺ繝ｻ・ｫ驛｢譎｢・ｽ・ｬ驛｢譎冗函・取刮・ｹ・ｧ繝ｻ・｢驛｢譏ｴ繝ｻ郢晢ｽｻ郢晢ｽｻ郢晢ｽｻ
    if state.total_lines_cleared // LINES_PER_LEVEL > state.last_total_lines_cleared // LINES_PER_LEVEL:
        print(f"Level Up! Total Lines Cleared: {state.total_lines_cleared}")
    
    # 髫ｴ蟠｢ﾂ髯溷供・ｾ蠕後・驛｢譎｢・ｽ・ｬ驛｢譎冗函・取刮・ｹ・ｧ繝ｻ・｢驛｢譏ｴ繝ｻ郢晢ｽｻ髫ｴ蠑ｱ・・ｹ晢ｽｻ驛｢譎｢・ｽ・ｩ驛｢・ｧ繝ｻ・､驛｢譎｢・ｽ・ｳ髫ｰ・ｨ繝ｻ・ｰ驛｢・ｧ陷ｻ莠･・ｳ・ｩ髫ｴ繝ｻ・ｽ・ｰ
    state.last_total_lines_cleared = state.total_lines_cleared

    # 髫ｴ繝ｻ・ｽ・ｰ驍ｵ・ｺ陷会ｽｱ繝ｻ讓抵ｽｹ譎・ｽｺ蛟･ﾎ驛｢・ｧ陜｣・､陷・ｽｽ髫ｰ蠕後・
    state.mino_pos = SPAWN_POS.copy()
    
    state.current_mino_type = state.next_mino_queue.pop(0)
    refill_next_queue()  # 驛｢譎樔ｺらｸｺ驢搾ｽｹ・ｧ繝ｻ・ｹ驛｢譎冗樟邵ｺ蜀暦ｽｹ譎｢・ｽ・･驛｢譎｢・ｽ・ｼ驛｢・ｧ髮区ｧｭ繝ｻ髯ｷ蛹ｻ繝ｻ繝ｻ・｡繝ｻ・ｫ
    state.next_mino_type = state.next_mino_queue[0] 
    
    state.mino = np.copy(minos[state.current_mino_type])
    state.can_hold = True  # 髫ｴ繝ｻ・ｽ・ｰ驍ｵ・ｺ陷会ｽｱ繝ｻ讓抵ｽｹ譎・ｽｺ蛟･ﾎ驍ｵ・ｺ隶吩ｸｻ繝ｻ髫ｰ迹壹・繝ｻ繝ｻ・ｹ・ｧ陟募ｨｯ陞ｺ驍ｵ・ｺ繝ｻ・ｮ驍ｵ・ｺ繝ｻ・ｧ驛｢譎擾ｽｸ蜷ｶ繝ｻ驛｢譎｢・ｽ・ｫ驛｢譎臥櫨陟弱・螯吶・・ｽ驍ｵ・ｺ繝ｻ・ｫ驍ｵ・ｺ陷ｷ・ｶ繝ｻ繝ｻ
    state.lock_timer = None  # 髫ｴ繝ｻ・ｽ・ｰ驍ｵ・ｺ陷会ｽｱ繝ｻ讓抵ｽｹ譎・ｽｺ蛟･ﾎ驍ｵ・ｺ隶吩ｸｻ繝ｻ髫ｰ迹壹・繝ｻ繝ｻ・ｹ・ｧ陟募ｨｯ陞ｺ驍ｵ・ｺ繝ｻ・ｮ驍ｵ・ｺ繝ｻ・ｧ驛｢譎｢・ｽ・ｭ驛｢譏ｴ繝ｻ邵ｺ驢搾ｽｹ・ｧ繝ｻ・ｿ驛｢・ｧ繝ｻ・､驛｢譎・ｽｧ・ｭ郢晢ｽｻ驛｢・ｧ陋幢ｽｵ・取㏍・ｹ・ｧ繝ｻ・ｻ驛｢譏ｴ繝ｻ郢晢ｽｨ
    state.lock_reset_counter = 0  # 驛｢譎｢・ｽ・ｭ驛｢譏ｴ繝ｻ邵ｺ驢搾ｽｹ譎｢・ｽ・ｪ驛｢・ｧ繝ｻ・ｻ驛｢譏ｴ繝ｻ郢晢ｽｨ驛｢・ｧ繝ｻ・ｫ驛｢・ｧ繝ｻ・ｦ驛｢譎｢・ｽ・ｳ驛｢・ｧ繝ｻ・ｿ驛｢譎｢・ｽ・ｼ驛｢・ｧ陋幢ｽｵ・取㏍・ｹ・ｧ繝ｻ・ｻ驛｢譏ｴ繝ｻ郢晢ｽｨ


    # 驛｢・ｧ繝ｻ・ｲ驛｢譎｢・ｽ・ｼ驛｢譎｢・｣・ｰ驛｢・ｧ繝ｻ・ｪ驛｢譎｢・ｽ・ｼ驛｢譎√・郢晢ｽｻ髯具ｽｻ繝ｻ・､髯橸ｽｳ郢晢ｽｻ
    if check_collision(state.mino_pos, state.mino):
        print("Game Over!")
        print(f"Final Score: {state.score}, Total Lines Cleared: {state.total_lines_cleared}")
        print("Press R to restart the game.")
        state.drop_switch = False
        state.game_over = True

def rotate_mino(direction):
    
    if state.current_mino_type == "O":
        return  # O驛｢譎・ｽｺ蛟･ﾎ驍ｵ・ｺ繝ｻ・ｯ髯懃軸・ｫ繝ｻ・ｽ・ｻ繝ｻ・｢驍ｵ・ｺ陷会ｽｱ遶企・・ｸ・ｺ郢晢ｽｻ
    
    new_mino = np.copy(state.mino)
    if direction == "left":
        for i in range(4):
            new_mino[i][0] = -state.mino[i][1]
            new_mino[i][1] = state.mino[i][0]
    elif direction == "right":
        for i in range(4):
            new_mino[i][0] = state.mino[i][1]
            new_mino[i][1] = -state.mino[i][0]
    else:
        return  # 髴取ｻゑｽｽ・｡髯ｷ莨夲ｽｽ・ｹ驍ｵ・ｺ繝ｻ・ｪ髫ｴ繝ｻ・ｽ・ｹ髯ｷ・ｷ闔会ｽ｣郢晢ｽｻ髯懶ｽ｣繝ｻ・ｴ髯ｷ・ｷ陋ｹ・ｻ郢晢ｽｻ髣厄ｽｴ髴郁ｲｻ・ｽ繧会ｽｸ・ｺ陷会ｽｱ遶企・・ｸ・ｺ郢晢ｽｻ
            
    rotated = False

    # 髯懃軸・ｫ繝ｻ・ｽ・ｻ繝ｻ・｢髯溷供・ｾ蠕後・髣厄ｽｴ陷･・ｲ繝ｻ・ｽ繝ｻ・ｮ驍ｵ・ｺ繝ｻ・ｧ鬮ｯ・ｦ隴惹ｼ夲ｽｽ・ｪ遶乗刋繝ｻ髯橸ｽｳ郢晢ｽｻ
    # 驛｢・ｧ繝ｻ・ｭ驛｢譏ｴ繝ｻ邵ｺ驢搾ｽｹ譏ｴ繝ｻ邵ｺ蟶ｷ・ｹ譎冗樟繝ｻ蟶晏惡繝ｻ・ｦ驍ｵ・ｺ郢晢ｽｻ
    for test in kick_tests:
        test_pos = state.mino_pos + np.array(test)
        if not check_collision(test_pos, new_mino):
            state.mino_pos = test_pos
            state.mino = new_mino
            rotated = True
            break  # 驛｢・ｧ繝ｻ・ｭ驛｢譏ｴ繝ｻ邵ｺ鬘鯉ｽｬ蠕｡・ｻ蜥擾ｽｲ・･驍ｵ・ｺ陷会ｽｱ隨ｳ繝ｻ・ｹ・ｧ陝ｲ・ｨ・取刮・ｹ譎｢・ｽ・ｼ驛｢譎丞ｹｲ繝ｻ螳夲ｽｬ螢ｽ繹ｱ繝ｻ・ｰ驛｢・ｧ郢晢ｽｻ
    
    if rotated and state.lock_reset_counter < MAX_LOCK_RESETS and state.lock_timer is not None:
        state.lock_timer = None  # 髯懃軸・ｫ繝ｻ・ｽ・ｻ繝ｻ・｢驍ｵ・ｺ陷会ｽｱ隨ｳ繝ｻ・ｸ・ｺ繝ｻ・ｮ驍ｵ・ｺ繝ｻ・ｧ驛｢譎｢・ｽ・ｭ驛｢譏ｴ繝ｻ邵ｺ驢搾ｽｹ・ｧ繝ｻ・ｿ驛｢・ｧ繝ｻ・､驛｢譎・ｽｧ・ｭ郢晢ｽｻ驛｢・ｧ陋幢ｽｵ・取㏍・ｹ・ｧ繝ｻ・ｻ驛｢譏ｴ繝ｻ郢晢ｽｨ
        state.lock_reset_counter += 1  # 驛｢譎｢・ｽ・ｪ驛｢・ｧ繝ｻ・ｻ驛｢譏ｴ繝ｻ郢晢ｽｨ髯懃軸・ｨ鬘斐・驛｢・ｧ陋幢ｽｵ邵ｺ蜥ｲ・ｹ・ｧ繝ｻ・ｦ驛｢譎｢・ｽ・ｳ驛｢譏ｴ繝ｻ


def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # 驛｢譎｢・ｽ・｢驛｢譏ｴ繝ｻ・取刮・ｹ譎∽ｾｭ・守､ｼ・ｹ譎｢・ｽ・ｼ髯樊ｺｽ蛻､鬩ｪ・､鬮ｯ・ｦ隰疲ｺ倥・驍ｵ・ｺ繝ｻ・ｮ鬮ｫ・ｪ繝ｻ・ｭ髯橸ｽｳ郢晢ｽｻ
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    gluLookAt(*CAMERA_EYE, *CAMERA_CENTER, *CAMERA_UP)
    
    glFrontFace(GL_CW) # 髫ｴ蠑ｱ・翫・・ｨ闔・･陞ｻ骰具ｽｹ・ｧ驗呻ｽｫ繝ｻ蟶晏距繝ｻ・ｨ鬯ｮ・ｱ繝ｻ・｢驍ｵ・ｺ繝ｻ・ｨ驍ｵ・ｺ陷ｷ・ｶ繝ｻ繝ｻ
    glCullFace(GL_BACK)
    
    glLightfv(GL_LIGHT0, GL_POSITION, LIGHT_POSITION)
    current_time = glfw.get_time()   
    
    # === 驛｢譎｢・ｽ・ｭ驛｢・ｧ繝ｻ・ｸ驛｢譏ｴ繝ｻ邵ｺ鬘梧弱・・ｦ鬨ｾ繝ｻ繝ｻ===
    if not state.game_over and state.drop_switch:
        fall_interval = max(
            MIN_FALL_INTERVAL,
            INITIAL_FALL_INTERVAL - (state.total_lines_cleared // LINES_PER_LEVEL) * FALL_INTERVAL_DECREASE_PER_LEVEL
        )

        # 鬯ｨ・ｾ陞｢・ｼ繝ｻ・ｸ繝ｻ・ｸ鬮｣諛ｶ・ｽ・ｽ髣包ｽｳ郢晢ｽｻ
        if current_time - state.last_drop_time > fall_interval:
            state.last_drop_time = current_time
            next_pos = state.mino_pos + np.array([0, -1, 0])

            if not check_collision(next_pos, state.mino):
                state.mino_pos = next_pos

        # 髫ｰ證ｦ・ｽ・･髯懶ｽｨ繝ｻ・ｰ髴托ｽ･繝ｻ・ｶ髫ｲ・ｷ闕ｵ譎｢・ｽ螳夲ｽｱ莠･・ｼ・ｱ郢晢ｽｵ驛｢譎｢・ｽ・ｬ驛｢譎｢・ｽ・ｼ驛｢譎｢・｣・ｰ鬩墓慣・ｽ・ｺ鬮ｫ・ｱ郢晢ｽｻ
        grounded = check_collision(
            state.mino_pos + np.array([0, -1, 0]),
            state.mino
        )

        if grounded:
            # 髫ｰ證ｦ・ｽ・･髯懶ｽｨ繝ｻ・ｰ驍ｵ・ｺ陷会ｽｱ隨ｳ繝ｻ・ｿ・ｸ繝ｻ・ｬ鬯ｮ・｢髦ｮ蜷ｮ繝ｻ髣包ｽｳ・つ髯溯ｶ｣・ｽ・ｦ驍ｵ・ｺ繝ｻ・ｰ驍ｵ・ｺ魄・ｽｹ陝ｷ謌頑ｲらｹ晢ｽｻ
            if state.lock_timer is None:
                state.lock_timer = current_time

            # 0.5鬩穂ｼ懶｣ｰ・､繝ｻ・ｵ驕停沖ﾑ・し・ｺ陷会ｽｱ隨ｳ繝ｻ・ｹ・ｧ霑壼雀・ｴ邇匁･懃ｹ晢ｽｻ
            elif current_time - state.lock_timer >= LOCK_DELAY:
                lock_and_spawn_mino()
        else:
            # 髫ｰ證ｦ・ｽ・･髯懶ｽｨ繝ｻ・ｰ髴托ｽ･繝ｻ・ｶ髫ｲ・ｷ闕ｵ謨厄ｽｰ驛｢・ｧ騾包ｽｻ隴ｬ・｢驍ｵ・ｺ闔会ｽ｣隨ｳ繝ｻ
            state.lock_timer = None



        # === 髫ｰ・ｰ陷諤懈・髯ｷ繝ｻ・ｽ・ｦ鬨ｾ繝ｻ繝ｻ===
    
    # 1. 髯懈圜・ｽ・ｺ髯橸ｽｳ陞｢・ｹ繝ｻ繝ｻ・ｹ・ｧ陟募ｨｯ陞ｺ驛｢譎・ｽｼ譁絶襖驛｢譎｢・ｽ・ｼ驛｢譎｢・ｽ・ｫ驛｢譎擾ｽｳ・ｨ繝ｻ螳夲ｽｬ・ｰ陷諤懈・    
    glPushMatrix()
    glTranslatef(*FLOOR_POS)
    drawPlaneY(FLOOR_SIZE)
    glPopMatrix()

    glPushMatrix()
    glTranslatef(*BACKGROUND_POS)
    drawPlaneZ(BACKGROUND_SIZE)
    glPopMatrix()

    glPushMatrix()
    drawField() 
    drawFrame()
    glPopMatrix()
    
    # 2. 驛｢譎樔ｺらｸｺ驢搾ｽｹ・ｧ繝ｻ・ｹ驛｢譎冗樟遶雁､・ｹ譎擾ｽｸ蜷ｶ繝ｻ驛｢譎｢・ｽ・ｫ驛｢譎擾ｽｳ・ｨ繝ｻ螳夲ｽｬ・ｰ陷諤懈・
    drawNextMinoPreview()

    glPushMatrix()
    glTranslatef(*HOLD_PREVIEW_POS)
    drawPreviewFrame()
    if state.hold_mino_type is not None:
        for i in range(4):
            drawCube(minos[state.hold_mino_type][i][0], minos[state.hold_mino_type][i][1], minos[state.hold_mino_type][i][2], COLORS[state.hold_mino_type])
    glPopMatrix()
        
    # 3. 髫ｰ・ｫ陜｣・ｺ繝ｻ・ｽ隲帑ｼ夲ｽｽ・ｸ繝ｻ・ｭ驍ｵ・ｺ繝ｻ・ｮ驛｢譎・ｽｺ蛟･ﾎ驍ｵ・ｺ繝ｻ・ｨ驛｢・ｧ繝ｻ・ｴ驛｢譎｢・ｽ・ｼ驛｢・ｧ繝ｻ・ｹ驛｢譎冗樟繝ｻ螳夲ｽｬ・ｰ陷諤懈・郢晢ｽｻ鬩包ｽｺ・つ繝ｻ・ｻ驛｢・ｧ繝ｻ・ｲ驛｢譎｢・ｽ・ｼ驛｢譎｢・｣・ｰ驛｢・ｧ繝ｻ・ｪ驛｢譎｢・ｽ・ｼ驛｢譎√・郢晢ｽｻ髫ｴ蠑ｱ・・ｹ晢ｽｻ髫ｰ・ｰ陷諤懈・驍ｵ・ｺ陷会ｽｱ遶企・・ｸ・ｺ郢晢ｽｻ繝ｻ・ｼ郢晢ｽｻ
    if not state.game_over:
        # 驛｢・ｧ繝ｻ・ｴ驛｢譎｢・ｽ・ｼ驛｢・ｧ繝ｻ・ｹ驛｢譎冗樟・朱・・ｹ譏ｴ繝ｻ
        ghost_pos = state.mino_pos.copy()
        # 驛｢・ｧ繝ｻ・ｴ驛｢譎｢・ｽ・ｼ驛｢・ｧ繝ｻ・ｹ驛｢譎冗樟・朱・・ｹ譎擾ｽｼ・ｱ郢晢ｽｻ髣厄ｽｴ陷･・ｲ繝ｻ・ｽ繝ｻ・ｮ驛｢・ｧ陞ｳ螟ｲ・ｽ・ｨ髢ｧ・ｲ繝ｻ・ｮ陷会ｽｱ隨倥・・ｹ・ｧ郢晢ｽｻ
        while not check_collision(ghost_pos + np.array([0, -1, 0]), state.mino):
            ghost_pos += np.array([0, -1, 0])

        if not np.array_equal(ghost_pos, state.mino_pos):
            for i in range(4):
                drawGhostCube(ghost_pos[0] + state.mino[i][0], ghost_pos[1] + state.mino[i][1], ghost_pos[2] + state.mino[i][2], COLORS[state.current_mino_type])

        # 髫ｰ・ｫ陜｣・ｺ繝ｻ・ｽ隲帑ｼ夲ｽｽ・ｸ繝ｻ・ｭ驛｢譎・ｽｺ蛟･ﾎ
        glPushMatrix()
        glTranslatef(state.mino_pos[0], state.mino_pos[1], state.mino_pos[2]) 
        for i in range(4):
            drawCube(state.mino[i][0], state.mino[i][1], state.mino[i][2], COLORS[state.current_mino_type]) 
        glPopMatrix()
        
    
    drawScorePreview()

        
    
# 驛｢譎・§・取ｺｽ・ｹ譏ｴ繝ｻ邵ｺ驢搾ｽｹ・ｧ陷ｻ閧ｲ・ｷ蟶敖蛹・ｽｽ・ｻ驍ｵ・ｺ陷ｷ・ｶ繝ｻ遏ｩ・ｫ・｢繝ｻ・｢髫ｰ・ｨ繝ｻ・ｰ
def drawCube(x=0.0, y=0.0, z=0.0, color=(1.0, 1.0, 1.0), texture_id=None):
    light_enabled = glGetBooleanv(GL_LIGHTING)

    glPushMatrix()
    glTranslatef(x, y, z)

    # 鬯ｮ・ｱ繝ｻ・｢驛｢・ｧ陷ｻ閧ｲ・ｷ蟶敖蛹・ｽｽ・ｻ
    glEnable(GL_LIGHTING)

    glMaterialfv(GL_FRONT, GL_SPECULAR, BLOCK_SPECULAR)
    glMaterialf(GL_FRONT, GL_SHININESS, BLOCK_SHININESS)

    for i in range(6):
        if state.game_over:
            glColor3fv((0.5, 0.5, 0.5))
        else:
            glColor3fv(color)

        glBegin(GL_QUADS)
        for j in range(4):
            glNormal3fv(normals[i])
            glVertex3fv(vertex[face[i][j]])
        glEnd()

    if texture_id is not None and not state.game_over:
        glDisable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glColor3f(1.0, 1.0, 1.0)

        # z+ 髯句ｹ｢・ｽ・ｴ驍ｵ・ｺ繝ｻ・ｮ鬯ｮ・ｱ繝ｻ・｢驍ｵ・ｺ繝ｻ・ｫ驍ｵ・ｲ遶丞｣ｺ諡ｬ驛｢・ｧ髦ｮ蜷ｶ繝ｻ髯昴・・ｻ・｣繝ｻ・ｰ髫ｰ繝ｻ蜚ｱ霎ｯ諷包ｽｸ・ｺ繝ｻ・ｸ驍ｵ・ｺ陞｢・ｹ繝ｻ閾･・ｸ・ｺ陷会ｽｱ遯ｶ・ｻ髫ｰ・ｨ繝ｻ・ｰ髯昴・蟷ｲ繝ｻ蟶晏ｯ槭・・ｼ驛｢・ｧ郢晢ｽｻ
        # 驍ｵ・ｺ陞｢・ｹ繝ｻ閾･・ｸ・ｺ髴域喚繝ｻ驍ｵ・ｺ郢晢ｽｻ遶雁､・ｹ・ｧ繝ｻ・ｭ驛｢譎｢・ｽ・･驛｢譎｢・ｽ・ｼ驛｢譎擾ｽ､諛亥ｳｶ驍ｵ・ｺ繝ｻ・ｨ髯ｷ・ｷ陟募具ｽｧ髮趣ｽｺ繝ｻ・ｱ髯溯ｶ｣・ｽ・ｦ驍ｵ・ｺ繝ｻ・ｫ驍ｵ・ｺ繝ｻ・ｪ驍ｵ・ｺ繝ｻ・｣驍ｵ・ｺ繝ｻ・ｦ驛｢譏ｶ繝ｻ・主ｸｷ・ｸ・ｺ繝ｻ・､驍ｵ・ｺ鬮ｦ・ｪ繝ｻ繝ｻ・ｸ・ｺ陷ｷ・ｶ繝ｻ繝ｻ
        offset = np.array([0.0, 0.0, 0.01])
        glBegin(GL_QUADS)
        glNormal3fv(normals[2])
        glTexCoord2f(1.0, 1.0)
        glVertex3fv(vertex[face[2][0]] + offset)
        glTexCoord2f(1.0, 0.0)
        glVertex3fv(vertex[face[2][1]] + offset)
        glTexCoord2f(0.0, 0.0)
        glVertex3fv(vertex[face[2][2]] + offset)
        glTexCoord2f(0.0, 1.0)
        glVertex3fv(vertex[face[2][3]] + offset)
        glEnd()

        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    # 鬮ｴ闌ｨ・ｽ・ｪ鬯ｩ蟷｢・ｽ・ｭ鬩搾ｽｱ陞｢・ｹ繝ｻ螳夲ｽｬ・ｰ陷諤懈・
    glDisable(GL_LIGHTING)
    glColor3f(*BLOCK_EDGE_COLOR)
    glLineWidth(BLOCK_EDGE_WIDTH)

    glBegin(GL_LINES)
    for i in range(12):
        glVertex3fv(vertex[edge[i][0]])
        glVertex3fv(vertex[edge[i][1]])
    glEnd()

    glPopMatrix()

    # 髯ｷ蛹ｻ繝ｻ郢晢ｽｻ驛｢譎｢・ｽ・ｩ驛｢・ｧ繝ｻ・､驛｢譏ｴ繝ｻ邵ｺ繝ｻ・ｹ譎｢・ｽ・ｳ驛｢・ｧ繝ｻ・ｰ髴托ｽ･繝ｻ・ｶ髫ｲ・ｷ闕ｵ譏ｶ繝ｻ髫ｰ魃会ｽｽ・ｻ驍ｵ・ｺ郢晢ｽｻ
    if light_enabled:
        glEnable(GL_LIGHTING)
    else:
        glDisable(GL_LIGHTING)

def drawNextMinoPreview():
    glPushMatrix()
    glTranslatef(*NEXT_PREVIEW_POS)
    
    for preview_index, mino_type in enumerate(state.next_mino_queue[:5]):
        glPushMatrix()
        glTranslatef(0.0, -preview_index * NEXT_PREVIEW_INTERVAL_Y, 0.0)

        drawPreviewFrame()

        for i in range(4):
            drawCube(
                minos[mino_type][i][0],
                minos[mino_type][i][1],
                minos[mino_type][i][2],
                COLORS[mino_type]
            )

        glPopMatrix()

    glPopMatrix()

# 驛｢・ｧ繝ｻ・ｴ驛｢譎｢・ｽ・ｼ驛｢・ｧ繝ｻ・ｹ驛｢譎冗樟郢晢ｽｶ驛｢譎｢・ｽ・ｭ驛｢譏ｴ繝ｻ邵ｺ驢搾ｽｹ・ｧ陷ｻ閧ｲ・ｷ蟶敖蛹・ｽｽ・ｻ驍ｵ・ｺ陷ｷ・ｶ繝ｻ遏ｩ・ｫ・｢繝ｻ・｢髫ｰ・ｨ繝ｻ・ｰ
# Draw state.score digits below the NEXT previews.
def drawScorePreview():
    score_digits = str(state.score)
    for i, digit_char in enumerate(score_digits):
        digit = int(digit_char)
        texture_id = digit_texture_id_dict.get(digit)
        drawCube(SCORE_POS[0] + i * DIGIT_SPACING, SCORE_POS[1], SCORE_POS[2], (1.0, 1.0, 1.0), texture_id)

def drawGhostCube(x=0.0, y=0.0, z=0.0, color=(1.0, 1.0, 1.0)):
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    light_enabled = glGetBooleanv(GL_LIGHTING)
    cull_enabled = glGetBooleanv(GL_CULL_FACE)

    # Ghost blocks should not hide the real mino or be hidden by face culling.
    glDepthMask(GL_FALSE)
    glDisable(GL_LIGHTING)
    glDisable(GL_CULL_FACE)

    glPushMatrix()
    glTranslatef(x, y, z)

    for i in range(6):
        glColor4f(color[0], color[1], color[2], GHOST_ALPHA)
        glBegin(GL_QUADS)
        for j in range(4):
            glNormal3fv(normals[i])
            glVertex3fv(vertex[face[i][j]])
        glEnd()

    glPopMatrix()

    glDepthMask(GL_TRUE)
    glDisable(GL_BLEND)

    if cull_enabled:
        glEnable(GL_CULL_FACE)
    if light_enabled:
        glEnable(GL_LIGHTING)

# 驛｢譎・ｽｼ譁絶襖驛｢譎｢・ｽ・ｼ驛｢譎｢・ｽ・ｫ驛｢譎擾ｽｳ・ｨ繝ｻ螳夲ｽｬ・ｰ陷諤懈・驍ｵ・ｺ陷ｷ・ｶ繝ｻ遏ｩ・ｫ・｢繝ｻ・｢髫ｰ・ｨ繝ｻ・ｰ
def drawField():
    for y in range(state.field.shape[0]):
        for x in range(state.field.shape[1]):
            val = state.field[y][x]
            if val != 0:
                drawCube(x, y, 0, ID_TO_COLOR[val])

# Y鬮ｴ繝ｻ・ｽ・ｸ髫ｴ繝ｻ・ｽ・ｹ髯ｷ・ｷ闔会ｽ｣郢晢ｽｻ髯晢ｽｷ繝ｻ・ｳ鬯ｮ・ｱ繝ｻ・｢驛｢・ｧ陷ｻ閧ｲ・ｷ蟶敖蛹・ｽｽ・ｻ驍ｵ・ｺ陷ｷ・ｶ繝ｻ遏ｩ・ｫ・｢繝ｻ・｢髫ｰ・ｨ繝ｻ・ｰ
def drawPlaneY(size):
    glBegin(GL_QUADS)
    glNormal3f(0.0, 1.0, 0.0)
    for i in range(size):
        for j in range(size):
            x = i - 0.5 * size
            z = j - 0.5 * size
            if (i + j) % 2 == 0:
                glColor3f(1, 1, 1)
            else:
                glColor3f(0.2, 0.2, 0.2)
                
            glVertex3f(x, 0, z)
            glVertex3f(x + 1, 0, z)
            glVertex3f(x + 1, 0, z + 1)
            glVertex3f(x, 0, z + 1)
    glEnd()

# Z鬮ｴ繝ｻ・ｽ・ｸ髫ｴ繝ｻ・ｽ・ｹ髯ｷ・ｷ闔会ｽ｣郢晢ｽｻ髯晢ｽｷ繝ｻ・ｳ鬯ｮ・ｱ繝ｻ・｢驛｢・ｧ陷ｻ閧ｲ・ｷ蟶敖蛹・ｽｽ・ｻ驍ｵ・ｺ陷ｷ・ｶ繝ｻ遏ｩ・ｫ・｢繝ｻ・｢髫ｰ・ｨ繝ｻ・ｰ
def drawPlaneZ(size):
    
    glBegin(GL_QUADS)
    glNormal3f(0.0, 0.0, 0.5)
    for i in range(size):
        for j in range(size):
            x = i - 0.5 * size
            y = j - 0.5 * size
            if (i + j) % 2 == 0:
                if state.game_over:
                    glColor3f(0.0, 0.0, 0.0)  # 髴難ｽ｣繝ｻ・ｰ雎ｼ・ｶ繝ｻ・ｲ
                else:    
                    glColor3f(0.0, 0.0, 0.0)  # 髫ｴ荳橸ｽｼ・ｱ繝ｻ迢暦ｽｸ・ｺ郢晢ｽｻ郢晢ｽｻ雎ｼ・ｶ繝ｻ・ｲ
            else:
                glColor3f(0.2, 0.2, 0.2)
                
            glVertex3f(x,     y,     0)
            glVertex3f(x,     y + 1, 0)
            glVertex3f(x + 1, y + 1, 0)
            glVertex3f(x + 1, y,     0)
    glEnd()

# 驛｢譎・ｽｼ驥・ｨ抵ｽｹ譎｢・ｽ・ｼ驛｢譎｢・｣・ｰ驛｢・ｧ陷ｻ閧ｲ・ｷ蟶敖蛹・ｽｽ・ｻ驍ｵ・ｺ陷ｷ・ｶ繝ｻ遏ｩ・ｫ・｢繝ｻ・｢髫ｰ・ｨ繝ｻ・ｰ
def drawFrame():
    glDisable(GL_LIGHTING) # 驛｢譎｢・ｽ・ｩ驛｢・ｧ繝ｻ・､驛｢譏ｴ繝ｻ邵ｺ繝ｻ・ｹ譎｢・ｽ・ｳ驛｢・ｧ繝ｻ・ｰ驛｢・ｧ陜｣・､隨冗霜諤上・・ｹ驍ｵ・ｺ繝ｻ・ｫ驍ｵ・ｺ陷会ｽｱ遯ｶ・ｻ鬩墓慣・ｽ・ｺ髯橸ｽｳ雋・ｪ繝ｻ雎ｼ・ｶ繝ｻ・ｲ驛｢・ｧ髮区ｧｭ繝ｻ驍ｵ・ｺ郢晢ｽｻ
    z = FIELD_FRAME_Z # 驛｢譎・§・取ｺｽ・ｹ譏ｴ繝ｻ邵ｺ驢搾ｽｸ・ｺ繝ｻ・ｮ髯ｷ螟ｧ豸ｵ隰ｫ繝ｻ1.0)驛｢・ｧ陋ｹ・ｻ繝ｻ鬘費ｽｸ・ｺ繝ｻ・ｻ驛｢・ｧ髦ｮ蜷ｶ繝ｻ髯昴・・ｻ・｣繝ｻ・ｰ髫ｰ繝ｻ蜚ｱ霎ｯ諷包ｽｸ・ｺ繝ｻ・ｫ髫ｰ・ｰ闕ｳ螂・ｽｿ・･

    glColor3f(*FIELD_GRID_COLOR)  # 驛｢譎・ｽｼ驥・ｨ抵ｽｹ譎｢・ｽ・ｼ驛｢譎｢・｣・ｰ驍ｵ・ｺ繝ｻ・ｮ雎ｼ・ｶ繝ｻ・ｲ驛｢・ｧ陞ｳ螟ｲ・ｽ・ｨ繝ｻ・ｭ髯橸ｽｳ郢晢ｽｻ

    # 髯ｷﾂ郢晢ｽｻ郢晢ｽｻ驍ｵ・ｺ繝ｻ・ｮ驛｢・ｧ繝ｻ・ｰ驛｢譎｢・ｽ・ｪ驛｢譏ｴ繝ｻ郢晢ｽｩ驛｢・ｧ陝ｶ謨鳴陞｢・ｼ繝ｻ・ｸ繝ｻ・ｸ驍ｵ・ｺ繝ｻ・ｮ髯樊ｻゑｽｽ・ｪ驍ｵ・ｺ髴域鱒ﾂ螳夲ｽｬ・ｰ陷諤懈・
    glLineWidth(FIELD_GRID_WIDTH)
    glBegin(GL_LINES)
    for x in range(1, FIELD_WIDTH):
        glVertex3f(x, 0, z)
        glVertex3f(x, FIELD_HEIGHT, z)

    for y in range(1, FIELD_HEIGHT):
        glVertex3f(0, y, z)
        glVertex3f(FIELD_WIDTH, y, z)
    glEnd()

    # 髯樊ｺｷ謌溯ｭｽ・ｧ驍ｵ・ｺ繝ｻ・ｰ驍ｵ・ｺ闔会ｽ｣繝ｻ螳壽｣斐・・ｪ驍ｵ・ｺ闕ｵ遉ｼ・ｷ蟶敖蛹・ｽｽ・ｻ
    glLineWidth(FIELD_OUTLINE_WIDTH)
    glBegin(GL_LINE_LOOP)
    glVertex3f(0, 0, z)
    glVertex3f(FIELD_WIDTH, 0, z)
    glVertex3f(FIELD_WIDTH, FIELD_HEIGHT, z)
    glVertex3f(0, FIELD_HEIGHT, z)
    glEnd()

    # 髯溷｢難ｽｪ雜｣・ｽ・ｶ陞｢・ｹ郢晢ｽｻ髫ｰ・ｰ陷諤懈・驍ｵ・ｺ繝ｻ・ｸ鬩搾ｽｱ陞｢・ｼ繝ｻ・ｹ郢晢ｽｻ繝ｻ螳夲ｽｰ・ｿ闕ｵ譎｢・ｼ繝ｻ・ｸ・ｺ繝ｻ・ｪ驍ｵ・ｺ郢晢ｽｻ
    glLineWidth(1.0)
    glEnable(GL_LIGHTING) # 驛｢譎｢・ｽ・ｩ驛｢・ｧ繝ｻ・､驛｢譏ｴ繝ｻ邵ｺ繝ｻ・ｹ譎｢・ｽ・ｳ驛｢・ｧ繝ｻ・ｰ驛｢・ｧ髮区ｧｭ繝ｻ驍ｵ・ｺ繝ｻ・ｫ髫ｰ魃会ｽｽ・ｻ驍ｵ・ｺ郢晢ｽｻ

# NEXT/HOLD驛｢譎丞ｹｲ・取ｨ抵ｽｹ譎∽ｾｭ・守､ｼ・ｹ譎｢・ｽ・ｼ驍ｵ・ｺ繝ｻ・ｮ髯樊ｺｷ謌溯ｭｽ・ｧ驛｢・ｧ陷ｻ閧ｲ・ｷ蟶敖蛹・ｽｽ・ｻ驍ｵ・ｺ陷ｷ・ｶ繝ｻ遏ｩ・ｫ・｢繝ｻ・｢髫ｰ・ｨ繝ｻ・ｰ
def drawPreviewFrame():
    light_enabled = glIsEnabled(GL_LIGHTING)
    glDisable(GL_LIGHTING)

    # 髯ｷ闌ｨ・ｽ・ｨ鬩墓ｩｸ・ｽ・ｮ鬯ｯ菫ｶ・ｧ・ｭ郢晢ｽｻ驛｢譎・ｽｺ蛟･ﾎ驍ｵ・ｺ隰疲ｺｷ・ｺ・ｶ驍ｵ・ｺ繝ｻ・ｾ驛｢・ｧ郢晢ｽｻx4驍ｵ・ｺ繝ｻ・ｮ髫ｴ・ｫ繝ｻ・ｰ
    left, right = PREVIEW_FRAME_LEFT, PREVIEW_FRAME_RIGHT
    bottom, top = PREVIEW_FRAME_BOTTOM, PREVIEW_FRAME_TOP
    z = PREVIEW_FRAME_Z

    glColor3f(1.0, 1.0, 1.0)
    glLineWidth(3.0)
    glBegin(GL_LINE_LOOP)
    glVertex3f(left, bottom, z)
    glVertex3f(right, bottom, z)
    glVertex3f(right, top, z)
    glVertex3f(left, top, z)
    glEnd()

    glLineWidth(1.0)
    if light_enabled:
        glEnable(GL_LIGHTING)

# 鬮ｯ・ｦ隴惹ｼ夲ｽｽ・ｪ遶乗刋繝ｻ髯橸ｽｳ陞溷・謔ｴ髫ｰ・ｨ繝ｻ・ｰ
def check_collision(next_pos, next_mino):
    # 4驍ｵ・ｺ繝ｻ・､驍ｵ・ｺ繝ｻ・ｮ驛｢譎・§・取ｺｽ・ｹ譏ｴ繝ｻ邵ｺ驢搾ｽｸ・ｺ隴擾ｽｴ繝ｻ讙趣ｽｸ・ｺ隶抵ｽｭ繝ｻ讙趣ｽｸ・ｺ繝ｻ・ｫ驍ｵ・ｺ繝ｻ・､驍ｵ・ｺ郢晢ｽｻ遯ｶ・ｻ鬮ｫ・ｱ繝ｻ・ｿ驍ｵ・ｺ繝ｻ・ｹ驛｢・ｧ郢晢ｽｻ
    for i in range(4):
        # 1. 驛｢譎・ｽｼ譁絶襖驛｢譎｢・ｽ・ｼ驛｢譎｢・ｽ・ｫ驛｢譎擾ｽｳ・ｨ郢晢ｽｻ鬩搾ｽｨ繝ｻ・ｶ髯昴・・ｽ・ｾ髯溯ｶ｣・ｽ・ｧ髫ｶ轣倡函繝ｻ蟶晏搦髢ｧ・ｲ繝ｻ・ｮ陷会ｽｱ隨倥・・ｹ・ｧ陷茨ｽｷ繝ｻ・ｼ闔・･雋よ・・ｲ繝ｻ・ｴ蜈ｷ・ｽ・ｽ陷･・ｲ繝ｻ・ｽ繝ｻ・ｮ 郢晢ｽｻ郢晢ｽｻ鬨ｾ・ｶ繝ｻ・ｸ髯昴・・ｽ・ｾ髣厄ｽｴ陷･・ｲ繝ｻ・ｽ繝ｻ・ｮ郢晢ｽｻ郢晢ｽｻ
        x = int(next_pos[0] + next_mino[i][0])
        y = int(next_pos[1] + next_mino[i][1])

        # 2. 髯橸ｽ｢遶丞､ｲ・ｽ繝ｻ・ｰ螳茨ｽｿ・ｫ繝ｻ蟶昴・遶丞｣ｺﾂ・ｳ髫ｰ螢ｽ繹ｱ繝ｻ・ｰ驍ｵ・ｺ繝ｻ・ｦ驍ｵ・ｺ郢晢ｽｻ遶企・・ｸ・ｺ郢晢ｽｻ・ゑｽｰ驛｢譏ｶ繝ｻ邵ｺ閾･・ｹ譏ｴ繝ｻ邵ｺ繝ｻ
        if x < 0 or x >= FIELD_WIDTH or y < 0:
            return True # 鬮ｯ・ｦ隴惹ｼ夲ｽｽ・ｪ遶丞､ｲ・ｼ・ｰ驍ｵ・ｺ雋翫ｑ・ｽ・ｼ郢晢ｽｻ

        # 3. 驍ｵ・ｺ陷ｷ・ｶ邵ｲ蝣､・ｸ・ｺ繝ｻ・ｫ驛｢譎・ｽｼ譁絶襖驛｢譎｢・ｽ・ｼ驛｢譎｢・ｽ・ｫ驛｢譎擾ｽｳ・ｨ遶頑･｢鞫弱・・ｺ髯橸ｽｳ陞｢・ｹ繝ｻ繝ｻ・ｹ・ｧ陟募ｨｯﾂ・ｻ驍ｵ・ｺ郢晢ｽｻ繝ｻ迢暦ｽｹ譎・§・取ｺｽ・ｹ譏ｴ繝ｻ邵ｺ驢搾ｽｸ・ｺ繝ｻ・ｨ鬯ｩ・･鬮ｦ・ｪ遶企・・ｸ・ｺ繝ｻ・｣驍ｵ・ｺ繝ｻ・ｦ驍ｵ・ｺ郢晢ｽｻ遶企・・ｸ・ｺ郢晢ｽｻ・ゑｽｰ驛｢譏ｶ繝ｻ邵ｺ閾･・ｹ譏ｴ繝ｻ邵ｺ繝ｻ
        if 0 <= y < state.field.shape[0] and state.field[y][x] != 0:
            return True # 鬮ｯ・ｦ隴惹ｼ夲ｽｽ・ｪ遶丞､ｲ・ｼ・ｰ驍ｵ・ｺ雋翫ｑ・ｽ・ｼ郢晢ｽｻ
            
    return False # 驍ｵ・ｺ繝ｻ・ｩ驍ｵ・ｺ髦ｮ蜷ｮ繝ｻ驛｢・ｧ郢ｧ螂・ｽｽ・｡隴惹ｼ夲ｽｽ・ｪ遶丞､ｲ・ｼ・ｰ驍ｵ・ｺ繝ｻ・ｪ驍ｵ・ｺ闕ｵ譏ｶ螟｢驍ｵ・ｺ雋翫ｑ・ｽ・ｼ髢ｧ・ｲ繝ｻ・ｧ繝ｻ・ｻ髯ｷ閧ｴ・｡・ｧK郢晢ｽｻ郢晢ｽｻ繝ｻ・ｼ郢晢ｽｻ

# 驛｢譎｢・ｽ・ｩ驛｢・ｧ繝ｻ・､驛｢譎｢・ｽ・ｳ髮趣ｽｸ闔・･隰碑・・ｹ・ｧ陋幢ｽｵ郢晢ｽ｡驛｢・ｧ繝ｻ・ｧ驛｢譏ｴ繝ｻ邵ｺ驢搾ｽｸ・ｺ陷ｷ・ｶ繝ｻ遏ｩ・ｫ・｢繝ｻ・｢髫ｰ・ｨ繝ｻ・ｰ
def check_line_clear():
    lines_cleared = 0
    # 驛｢譎｢・ｽ・ｩ驛｢・ｧ繝ｻ・､驛｢譎｢・ｽ・ｳ髮趣ｽｸ闔・･隰碑・・ｹ・ｧ繝ｻ・｢驛｢譎｢・ｽ・ｫ驛｢・ｧ繝ｻ・ｴ驛｢譎｢・ｽ・ｪ驛｢・ｧ繝ｻ・ｺ驛｢譎｢・｣・ｰ
    new_field = np.zeros((FIELD_HEIGHT, FIELD_WIDTH), dtype=int)
    new_y = 0
    
    for y in range(FIELD_HEIGHT):
        if np.all(state.field[y] != 0): # 驍ｵ・ｺ隴擾ｽｴ郢晢ｽｻ鬮ｯ・ｦ陟募ｨｯﾂ・ｲ髯ｷ闌ｨ・ｽ・ｨ驍ｵ・ｺ繝ｻ・ｦ髯懆ｬ趣ｽｹ譏ｶ遨宣し・ｺ繝ｻ・｣驍ｵ・ｺ繝ｻ・ｦ驍ｵ・ｺ郢晢ｽｻ繝ｻ迢暦ｽｸ・ｺ郢晢ｽｻ
            lines_cleared += 1
            print(f"Line {y} cleared!")
        else:
            # 髯懆ｬ趣ｽｹ譏ｶ遨宣し・ｺ繝ｻ・｣驍ｵ・ｺ繝ｻ・ｦ驍ｵ・ｺ郢晢ｽｻ遶企・・ｸ・ｺ郢晢ｽｻ繝ｻ・｡陟募ｨｯ蜻ｳ驍ｵ・ｺ闔会ｽ｣繝ｻ螳夲ｽｭ繝ｻ・ｽ・ｰ驍ｵ・ｺ陷会ｽｱ繝ｻ讓抵ｽｹ譎・ｽｼ譁絶襖驛｢譎｢・ｽ・ｼ驛｢譎｢・ｽ・ｫ驛｢譎擾ｽｳ・ｨ遶企ｦｴ蝨ｦ繝ｻ・ｰ驛｢・ｧ遶丞｣ｺﾂ・ｻ驍ｵ・ｺ郢晢ｽｻ繝ｻ・･
            new_field[new_y] = state.field[y]
            new_y += 1
            
    state.field = new_field # 驛｢譎・ｽｼ譁絶襖驛｢譎｢・ｽ・ｼ驛｢譎｢・ｽ・ｫ驛｢譎擾ｽｳ・ｨ繝ｻ螳壼初鬯・､ｧ・ｶ讙趣ｽｸ・ｺ郢晢ｽｻ
    return lines_cleared

# 驛｢・ｧ繝ｻ・ｭ驛｢譎｢・ｽ・ｼ驛｢譎・鯵郢晢ｽｻ驛｢譎臥櫨郢晢ｽｻ髯ｷ迚呻ｽｸ蜷ｶ繝ｻ驛｢・ｧ繝ｻ・ｳ驛｢譎｢・ｽ・ｼ驛｢譎｢・ｽ・ｫ驛｢譎√・郢晢ｽ｣驛｢・ｧ繝ｻ・ｯ驛｢・ｧ陜｣・､陋ｹ・ｳ鬯ｪ・ｭ繝ｻ・ｲ
def keyboard(window, key, scancode, action, mods):

    # RELEASE驛｢・ｧ繝ｻ・､驛｢譎冗函・趣ｽｦ驛｢譎冗樟郢晢ｽｻ髣厄ｽｴ繝ｻ・ｿ鬨ｾ蛹・ｽｽ・ｨ驍ｵ・ｺ陷会ｽｱ遶企・・ｸ・ｺ郢晢ｽｻ
    if action not in (glfw.PRESS, glfw.REPEAT):
        return

    # 鬩搾ｽｨ郢ｧ繝ｻ・ｽ・ｺ郢晢ｽｻ遶雁､・ｹ譎｢・ｽ・ｪ驛｢・ｧ繝ｻ・ｹ驛｢・ｧ繝ｻ・ｿ驛｢譎｢・ｽ・ｼ驛｢譎冗樟郢晢ｽｻ髯晢ｽｶ繝ｻ・ｸ驍ｵ・ｺ繝ｻ・ｫ髯ｷ・ｿ陷会ｽｱ繝ｻ・ｰ髣皮甥ﾂ・･繝ｻ・ｰ驛｢・ｧ郢晢ｽｻ
    if action == glfw.PRESS and key == KEY_QUIT:
        glfw.set_window_should_close(window, True)
        print("Quit key pressed - exiting")
        return

    if action == glfw.PRESS and key == KEY_RESTART:
        reset_game()
        return

    if state.game_over:
        return

    # P驍ｵ・ｺ繝ｻ・ｧ驛｢譎・ｺ｢郢晢ｽｻ驛｢・ｧ繝ｻ・ｺ郢晢ｽｻ闕ｳ讒ｭ繝ｻ鬯ｮ・｢闕ｵ謨鳴郢ｧ繝ｻ驟ｪ髮弱・・ｽ・｢髣包ｽｳ繝ｻ・ｭ驍ｵ・ｺ繝ｻ・ｮ髫ｴ蠑ｱ・玖将・｣驛｢・ｧ髮区ｨ願ｳ驛｢・ｧ繝ｻ・ｿ驛｢・ｧ繝ｻ・､驛｢譎・ｽｧ・ｭ郢晢ｽｻ驍ｵ・ｺ闕ｵ譎｢・ｽ陋ｾ・ｫ・ｯ繝ｻ・､髯樊ｻ薙§隨倥・・ｹ・ｧ郢晢ｽｻ
    if action == glfw.PRESS and key == KEY_PAUSE:
        current_time = glfw.get_time()
        if state.drop_switch:
            state.drop_switch = False
            state.pause_started_time = current_time
            print("Paused")
        else:
            if state.pause_started_time is not None:
                paused_duration = current_time - state.pause_started_time
                state.last_drop_time += paused_duration
                if state.lock_timer is not None:
                    state.lock_timer += paused_duration
            state.drop_switch = True
            state.pause_started_time = None
            print("Resumed")
        refresh(window)
        return

    # 驛｢譎・ｺ｢郢晢ｽｻ驛｢・ｧ繝ｻ・ｺ髣包ｽｳ繝ｻ・ｭ驍ｵ・ｺ繝ｻ・ｯ驛｢・ｧ繝ｻ・ｲ驛｢譎｢・ｽ・ｼ驛｢譎｢・｣・ｰ髫ｰ・ｫ陜｣・ｺ繝ｻ・ｽ隲帛･・ｽｽ螳壽╂陷会ｽｱ繝ｻ・ｰ髣皮甥ﾂ・･繝ｻ・ｰ驍ｵ・ｺ繝ｻ・ｪ驍ｵ・ｺ郢晢ｽｻ
    if not state.drop_switch:
        return

    # 髯晢ｽｾ繝ｻ・ｦ髯ｷ・ｿ繝ｻ・ｳ鬩募∞・ｽ・ｻ髯ｷ讎頑｡√・・ｼ陜捺ｻ難ｽｬ・ｾ驍ｵ・ｺ陷会ｽｱ隨・ｽｲ驍ｵ・ｺ繝ｻ・ｱ驍ｵ・ｺ繝ｻ・ｪ驍ｵ・ｺ隲､諛ｶ・ｽ・ｯ繝ｻ・ｾ髯滂ｽ｢隲幢ｽｶ繝ｻ・ｼ郢晢ｽｻ
    if key == KEY_MOVE_RIGHT:
        if not check_collision(state.mino_pos + np.array([1, 0, 0]), state.mino):
            state.mino_pos[0] += 1
            if state.lock_reset_counter < MAX_LOCK_RESETS and state.lock_timer is not None:
                state.lock_timer = None
                state.lock_reset_counter += 1

    elif key == KEY_MOVE_LEFT:
        if not check_collision(state.mino_pos + np.array([-1, 0, 0]), state.mino):
            state.mino_pos[0] -= 1
            if state.lock_reset_counter < MAX_LOCK_RESETS and state.lock_timer is not None:
                state.lock_timer = None
                state.lock_reset_counter += 1

    # 驕ｶ莨∝ｱｮ繝ｻ・ｼ陞｢・ｹ邵ｺ貅ｽ・ｹ譎・ｽｼ譁石夐Δ譎擾ｽｳ・ｨ・取ｺｽ・ｹ譏ｴ繝ｻ郢晢ｽｻ郢晢ｽｻ陜捺ｻ難ｽｬ・ｾ驍ｵ・ｺ陷会ｽｱ隨・ｽｲ驍ｵ・ｺ繝ｻ・ｱ驍ｵ・ｺ繝ｻ・ｪ驍ｵ・ｺ隲､諛ｶ・ｽ・ｯ繝ｻ・ｾ髯滂ｽ｢隲幢ｽｶ繝ｻ・ｼ郢晢ｽｻ
    elif key == KEY_SOFT_DROP:
        next_pos = state.mino_pos + np.array([0, -1, 0])
        if not check_collision(next_pos, state.mino):
            state.mino_pos = next_pos
            state.last_drop_time = glfw.get_time()
            state.lock_timer = None

    # Space郢晢ｽｻ陞｢・ｹ郢晢ｽｯ驛｢譎｢・ｽ・ｼ驛｢譎擾ｽｳ・ｨ郢晢ｽｩ驛｢譎｢・ｽ・ｭ驛｢譏ｴ繝ｻ郢晢ｽｻ
    elif action == glfw.PRESS and key == KEY_HARD_DROP:
        while not check_collision(state.mino_pos + np.array([0, -1, 0]), state.mino):
            state.mino_pos[1] -= 1
        lock_and_spawn_mino()
        state.last_drop_time = glfw.get_time()

    # 驕ｶ鬆第ｱ壹・・ｼ鬩｢諛翫・陞｢・ｼ隰・ｽｿ髯懃軸・ｫ繝ｻ・ｽ・ｻ繝ｻ・｢
    elif action == glfw.PRESS and key in KEY_ROTATE_RIGHT:
        rotate_mino("right")

    # Z郢晢ｽｻ髢ｾ・ｭeft Ctrl郢晢ｽｻ陞｢・ｼ繝ｻ・ｷ繝ｻ・ｦ髯懃軸・ｫ繝ｻ・ｽ・ｻ繝ｻ・｢
    elif action == glfw.PRESS and key in KEY_ROTATE_LEFT:
        rotate_mino("left")

    # C郢晢ｽｻ髢ｾ・ｭeft Shift郢晢ｽｻ陞｢・ｹ郢晢ｽｻ驛｢譎｢・ｽ・ｼ驛｢譎｢・ｽ・ｫ驛｢譏ｴ繝ｻ
    elif action == glfw.PRESS and key in KEY_HOLD:
        if state.can_hold:
            if state.hold_mino_type is None:
                state.hold_mino_type = state.current_mino_type
                state.current_mino_type = state.next_mino_queue.pop(0)
                refill_next_queue()  # 驛｢譎樔ｺらｸｺ驢搾ｽｹ・ｧ繝ｻ・ｹ驛｢譎冗樟邵ｺ蜀暦ｽｹ譎｢・ｽ・･驛｢譎｢・ｽ・ｼ驛｢・ｧ髮区ｧｭ繝ｻ髯ｷ蛹ｻ繝ｻ繝ｻ・｡繝ｻ・ｫ
                state.next_mino_type = state.next_mino_queue[0]  # 驛｢譎樔ｺらｸｺ驢搾ｽｹ・ｧ繝ｻ・ｹ驛｢譎冗樟邵ｺ蜀暦ｽｹ譎｢・ｽ・･驛｢譎｢・ｽ・ｼ驍ｵ・ｺ繝ｻ・ｮ髯ｷ閧ｲ・｣・ｯ繝ｻ・ｰ繝ｻ・ｭ驛｢・ｧ陷ｻ闌ｨ・ｽ・ｬ繝ｻ・｡驍ｵ・ｺ繝ｻ・ｮ驛｢譎・ｽｺ蛟･ﾎ驍ｵ・ｺ繝ｻ・ｨ驍ｵ・ｺ陷会ｽｱ遯ｶ・ｻ鬮ｫ・ｪ繝ｻ・ｭ髯橸ｽｳ郢晢ｽｻ
                
            else:
                state.current_mino_type, state.hold_mino_type = state.hold_mino_type, state.current_mino_type

            state.mino = np.copy(minos[state.current_mino_type])
            state.mino_pos = SPAWN_POS.copy()
            
            if check_collision(state.mino_pos, state.mino):
                print("Game Over!")
                print(f"Final Score: {state.score}, Total Lines Cleared: {state.total_lines_cleared}")
                print("Press R to restart the game.")
                state.drop_switch = False
                state.game_over = True
                return
            
            state.lock_timer = None
            state.lock_reset_counter = 0
            state.last_drop_time = glfw.get_time()
            state.can_hold = False
    
def refresh(window):
    display()
    glfw.swap_buffers(window)
    
def perspective(width, height):
    # 鬯ｨ・ｾ陷托ｽｰ繝ｻ・ｦ鬮｢ﾂ繝ｻ・､騾包ｽｻ鬩ｪ・､鬮ｯ・ｦ隰疲ｺ倥・驍ｵ・ｺ繝ｻ・ｮ鬮ｫ・ｪ繝ｻ・ｭ髯橸ｽｳ郢晢ｽｻ
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, width / height, 1.0, 100.0)
    # 驛｢譎｢・ｽ・｢驛｢譏ｴ繝ｻ・取刮・ｹ譎∽ｾｭ・守､ｼ・ｹ譎｢・ｽ・ｼ髯樊ｺｽ蛻､鬩ｪ・､鬮ｯ・ｦ隰疲ｺ倥・驍ｵ・ｺ繝ｻ・ｮ鬮ｫ・ｪ繝ｻ・ｭ髯橸ｽｳ郢晢ｽｻ
    glMatrixMode(GL_MODELVIEW)
    
def resize(window, width, height):
    perspective(width, height)
    
def init():
    glClearColor(0.2, 0.2, 0.2, 1.0)
    perspective(WINDOW_WIDTH, WINDOW_WIDTH)
    
    glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    
    glEnable(GL_COLOR_MATERIAL) 
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
    
    # 驛｢・ｧ繝ｻ・ｲ驛｢譎｢・ｽ・ｼ驛｢譎｢・｣・ｰ髴托ｽ･繝ｻ・ｶ髫ｲ・ｷ闕ｵ譎｢・ｽ螳壼ｴ戊ｭ弱・・・刹・ｹ郢晢ｽｻ
    reset_game()
    
def main():
    global digit_texture_id_dict
    glfw.init()
    
    glfw.window_hint(glfw.SAMPLES, 4)
    
    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, None, None)
    glfw.make_context_current(window)
    init()

    digit_texture_id_dict = {
        i: initTextureFromFile(DIGIT_DIR / f"{i}.png")
        for i in range(10)
    }
    
    glfw.set_window_refresh_callback(window, refresh)
    glfw.set_key_callback(window, keyboard)
    
    refresh(window)
    
    while not glfw.window_should_close(window):
        glfw.poll_events()
        display()
        glfw.swap_buffers(window)
        
        
    if digit_texture_id_dict:
        glDeleteTextures([int(texture_id) for texture_id in digit_texture_id_dict.values()])
    glfw.destroy_window(window)
    glfw.terminate()

if __name__ == "__main__":
    main()

#==========================
# 驛｢・ｧ繝ｻ・ｭ驛｢譎｢・ｽ・ｼ鬮ｫ・ｪ繝ｻ・ｭ髯橸ｽｳ陞｢・ｻ繝ｻ・ｸ・つ鬮ｫ蛹・ｽｽ・ｧ
'''
驕ｶ鄙ｫ繝ｻ/ 驕ｶ鄙ｫ繝ｻ         |  髯晢ｽｾ繝ｻ・ｦ髯ｷ・ｿ繝ｻ・ｳ鬩募∞・ｽ・ｻ髯ｷ髦ｪ繝ｻ
驕ｶ鄙ｫ繝ｻ             |  驛｢・ｧ繝ｻ・ｽ驛｢譎・ｽｼ譁石夐Δ譎擾ｽｳ・ｨ・取ｺｽ・ｹ譏ｴ繝ｻ郢晢ｽｻ
Space          |  驛｢譏懶ｽｸ鄙ｫ繝ｻ驛｢譎擾ｽｳ・ｨ郢晢ｽｩ驛｢譎｢・ｽ・ｭ驛｢譏ｴ繝ｻ郢晢ｽｻ
驕ｶ鄙ｫ繝ｻ/ E          |  髯ｷ・ｿ繝ｻ・ｳ髯懃軸・ｫ繝ｻ・ｽ・ｻ繝ｻ・｢
Q / Left Ctrl  |  髯晢ｽｾ繝ｻ・ｦ髯懃軸・ｫ繝ｻ・ｽ・ｻ繝ｻ・｢
C / Left Shift |  驛｢譎擾ｽｸ蜷ｶ繝ｻ驛｢譎｢・ｽ・ｫ驛｢譏ｴ繝ｻ
P              |  驛｢譎・ｺ｢郢晢ｽｻ驛｢・ｧ繝ｻ・ｺ/髯ｷﾂ陜難ｽｼ陝ｷ繝ｻ
R              |  驛｢譎｢・ｽ・ｪ驛｢・ｧ繝ｻ・ｹ驛｢・ｧ繝ｻ・ｿ驛｢譎｢・ｽ・ｼ驛｢譏ｴ繝ｻ
Escape         |  鬩搾ｽｨ郢ｧ繝ｻ・ｽ・ｺ郢晢ｽｻ
'''
#==========================
