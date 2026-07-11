# -*- coding: utf-8 -*-
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass, field as dataclass_field


# キューブの頂点座標
vertex = np.array([
( 0.0, 0.0, 0.0 ),
( 1.0, 0.0, 0.0 ),
( 1.0, 1.0, 0.0 ),
( 0.0, 1.0, 0.0 ),
( 0.0, 0.0, 1.0 ),
( 1.0, 0.0, 1.0 ),
( 1.0, 1.0, 1.0 ),
( 0.0, 1.0, 1.0 )
])

# キューブの辺を構成する頂点番号
edge = np.array([
[0, 1], [1, 2], [2, 3], [3, 0],
[4, 5], [5, 6], [6, 7], [7, 4],
[0, 4], [1, 5], [2, 6], [3, 7]
])

# キューブの各面を構成する頂点番号
face = np.array([
[3, 2, 1, 0],
[2, 1, 5, 6],
[6, 5, 4, 7],
[3, 7, 4, 0],
[0, 4, 5, 1],
[7, 3, 2, 6]
])

# 各面の法線ベクトル
normals = np.array([
[ 0.0, 0.0,-1.0],
[ 0.0, 1.0, 0.0],
[ 0.0, 0.0, 1.0],
[ 0.0,-1.0, 0.0],
[-1.0, 0.0, 0.0],
[ 1.0, 0.0, 0.0]
])

# 各ミノを構成する4ブロックの相対座標
minos = {
    "I": np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]]),
    "O": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
    "T": np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]]),
    "S": np.array([[0, 0, 0], [1, 0, 0], [-1, 1, 0], [0, 1, 0]]),
    "Z": np.array([[-1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]]),
    "J": np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0], [-1, 1, 0]]),
    "L": np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]])
}

# 回転時に試す簡易ウォールキック候補
kick_tests = [
    [0, 0, 0],
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [2, 0, 0],
    [-2, 0, 0],
]

# ミノごとの表示色
COLORS = {
    "I": (0.0, 1.0, 1.0),
    "O": (1.0, 1.0, 0.0),
    "T": (1.0, 0.0, 1.0),
    "S": (0.0, 1.0, 0.0),
    "Z": (1.0, 0.0, 0.0),
    "J": (0.0, 0.0, 1.0),
    "L": (1.0, 0.5, 0.0)
}

# フィールド保存用の色ID
COLOR_IDS = {"I": 1, "O": 2, "T": 3, "S": 4, "Z": 5, "J": 6, "L": 7}
# 色IDから表示色へ戻す表
ID_TO_COLOR = {1: COLORS["I"], 2: COLORS["O"], 3: COLORS["T"], 4: COLORS["S"], 5: COLORS["Z"], 6: COLORS["J"], 7: COLORS["L"]}

# ゲーム中に変化する状態をまとめるデータクラス
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
# フィールド設定
# ==========================

FIELD_WIDTH = 10
FIELD_HEIGHT = 20
SPAWN_POS = np.array([4.0, 19.0, 0.0])

# ==========================
# ゲームルール設定
# ==========================
NEXT_QUEUE_SIZE = 5
LOCK_DELAY = 0.5
MAX_LOCK_RESETS = 15
LINES_PER_LEVEL = 10
MIN_FALL_INTERVAL = 0.1
INITIAL_FALL_INTERVAL = 1.0
FALL_INTERVAL_DECREASE_PER_LEVEL = 0.1

# ==========================
# スコア設定
# ==========================
SCORE_TABLE = {
    1: 100,
    2: 300,
    3: 500,
    4: 800,
}

# ==========================
# ウィンドウ設定
# ==========================
WINDOW_WIDTH = 512
WINDOW_HEIGHT = 768
WINDOW_TITLE = "3D Tetris"

# ==========================
# カメラ・ライト設定
# ==========================
CAMERA_EYE = (15.0, 10.0, 30.0)
CAMERA_CENTER = (5.0, 10.0, 0.0)
CAMERA_UP = (0.0, 1.0, 0.0)

LIGHT_POSITION = [5.0, 10.0, 20.0, 1.0]

# ==========================
# 背景設定
# ==========================
FLOOR_SIZE = 12
BACKGROUND_SIZE = 22
FLOOR_POS = (5.0, 0.0, -1.0)
BACKGROUND_POS = (5.0, 10.0, -1.0)

# ==========================
# UI配置
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
# 描画スタイル
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
# アセットパス
# ==========================
BASE_DIR = Path(__file__).resolve().parent
DIGIT_DIR = BASE_DIR / "digits"

# ==========================
# キー設定
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

# 画像ファイルをOpenGLテクスチャとして読み込む
def initTextureFromFile(filename):

    with Image.open(filename) as img:
        image = img.convert("RGB")
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        width, height = image.size
        data = image.tobytes()

    texture_id = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, texture_id)

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

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_2D, 0)
    
    return texture_id

# ゲーム状態を初期化する
def reset_game():
    
    state.score = 0
    state.total_lines_cleared = 0
    state.last_total_lines_cleared = 0
    state.lock_reset_counter = 0
    state.lock_timer = None
    state.pause_started_time = None

    state.mino_bag.clear()
    state.current_mino_type = get_next_mino_type()
    state.next_mino_queue.clear()
    refill_next_queue()
    state.next_mino_type = state.next_mino_queue[0]
    state.hold_mino_type = None
    state.can_hold = True
    state.mino_pos = SPAWN_POS.copy()
    state.mino = np.copy(minos[state.current_mino_type])
    
    state.drop_switch = True
    state.last_drop_time = glfw.get_time()
    state.field = np.zeros((FIELD_HEIGHT, FIELD_WIDTH), dtype=int) 
    state.game_over = False
    print("Game Reset!")


# 7種バッグから次のミノを1つ取り出す
def get_next_mino_type():
    
    if not state.mino_bag:
        state.mino_bag.extend(list(minos.keys()))
        np.random.shuffle(state.mino_bag)
    return state.mino_bag.pop()

# NEXTキューが指定数になるまで補充する
def refill_next_queue():
    while len(state.next_mino_queue) < NEXT_QUEUE_SIZE:
        state.next_mino_queue.append(get_next_mino_type())
        
# 現在のミノを固定し、ライン消去後に次のミノを生成する
def lock_and_spawn_mino():

    if state.game_over:
        return

    color_id = COLOR_IDS[state.current_mino_type]
    for i in range(4):
        x = int(state.mino_pos[0] + state.mino[i][0])
        y = int(state.mino_pos[1] + state.mino[i][1])
        if 0 <= x < FIELD_WIDTH and 0 <= y < FIELD_HEIGHT and state.field[y][x] == 0:
            state.field[y][x] = color_id

    lines_cleared = check_line_clear()
    state.score += SCORE_TABLE.get(lines_cleared, 0)
    
    state.total_lines_cleared += lines_cleared
    
    if state.total_lines_cleared // LINES_PER_LEVEL > state.last_total_lines_cleared // LINES_PER_LEVEL:
        print(f"Level Up! Total Lines Cleared: {state.total_lines_cleared}")
    
    state.last_total_lines_cleared = state.total_lines_cleared

    state.mino_pos = SPAWN_POS.copy()
    
    state.current_mino_type = state.next_mino_queue.pop(0)
    refill_next_queue()
    state.next_mino_type = state.next_mino_queue[0] 
    
    state.mino = np.copy(minos[state.current_mino_type])
    state.can_hold = True
    state.lock_timer = None
    state.lock_reset_counter = 0


    if check_collision(state.mino_pos, state.mino):
        print("Game Over!")
        print(f"Final Score: {state.score}, Total Lines Cleared: {state.total_lines_cleared}")
        print("Press R to restart the game.")
        state.drop_switch = False
        state.game_over = True

# ミノを回転し、必要なら簡易ウォールキックを試す
def rotate_mino(direction):
    
    if state.current_mino_type == "O":
        return
    
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
        return
            
    rotated = False

    for test in kick_tests:
        test_pos = state.mino_pos + np.array(test)
        if not check_collision(test_pos, new_mino):
            state.mino_pos = test_pos
            state.mino = new_mino
            rotated = True
            break
    
    if rotated and state.lock_reset_counter < MAX_LOCK_RESETS and state.lock_timer is not None:
        state.lock_timer = None
        state.lock_reset_counter += 1


# 1フレーム分のゲーム更新と描画を行う
def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    gluLookAt(*CAMERA_EYE, *CAMERA_CENTER, *CAMERA_UP)
    
    glFrontFace(GL_CW)
    glCullFace(GL_BACK)
    
    glLightfv(GL_LIGHT0, GL_POSITION, LIGHT_POSITION)
    current_time = glfw.get_time()   
    
    if not state.game_over and state.drop_switch:
        fall_interval = max(
            MIN_FALL_INTERVAL,
            INITIAL_FALL_INTERVAL - (state.total_lines_cleared // LINES_PER_LEVEL) * FALL_INTERVAL_DECREASE_PER_LEVEL
        )

        if current_time - state.last_drop_time > fall_interval:
            state.last_drop_time = current_time
            next_pos = state.mino_pos + np.array([0, -1, 0])

            if not check_collision(next_pos, state.mino):
                state.mino_pos = next_pos

        grounded = check_collision(
            state.mino_pos + np.array([0, -1, 0]),
            state.mino
        )

        if grounded:
            if state.lock_timer is None:
                state.lock_timer = current_time

            elif current_time - state.lock_timer >= LOCK_DELAY:
                lock_and_spawn_mino()
        else:
            state.lock_timer = None



    
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
    
    drawNextMinoPreview()

    glPushMatrix()
    glTranslatef(*HOLD_PREVIEW_POS)
    drawPreviewFrame()
    if state.hold_mino_type is not None:
        for i in range(4):
            drawCube(minos[state.hold_mino_type][i][0], minos[state.hold_mino_type][i][1], minos[state.hold_mino_type][i][2], COLORS[state.hold_mino_type])
    glPopMatrix()
        
    if not state.game_over:
        ghost_pos = state.mino_pos.copy()
        while not check_collision(ghost_pos + np.array([0, -1, 0]), state.mino):
            ghost_pos += np.array([0, -1, 0])

        if not np.array_equal(ghost_pos, state.mino_pos):
            for i in range(4):
                drawGhostCube(ghost_pos[0] + state.mino[i][0], ghost_pos[1] + state.mino[i][1], ghost_pos[2] + state.mino[i][2], COLORS[state.current_mino_type])

        glPushMatrix()
        glTranslatef(state.mino_pos[0], state.mino_pos[1], state.mino_pos[2]) 
        for i in range(4):
            drawCube(state.mino[i][0], state.mino[i][1], state.mino[i][2], COLORS[state.current_mino_type]) 
        glPopMatrix()
        
    
    drawScorePreview()

        
    
# 1ブロックを描画する。texture_idがある場合は前面に数字テクスチャを貼る
def drawCube(x=0.0, y=0.0, z=0.0, color=(1.0, 1.0, 1.0), texture_id=None):
    light_enabled = glGetBooleanv(GL_LIGHTING)

    glPushMatrix()
    glTranslatef(x, y, z)

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

    glDisable(GL_LIGHTING)
    glColor3f(*BLOCK_EDGE_COLOR)
    glLineWidth(BLOCK_EDGE_WIDTH)

    glBegin(GL_LINES)
    for i in range(12):
        glVertex3fv(vertex[edge[i][0]])
        glVertex3fv(vertex[edge[i][1]])
    glEnd()

    glPopMatrix()

    if light_enabled:
        glEnable(GL_LIGHTING)
    else:
        glDisable(GL_LIGHTING)

# NEXTミノを最大5個表示する
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

# スコアを数字テクスチャ付きキューブで表示する
def drawScorePreview():
    score_digits = str(state.score)
    for i, digit_char in enumerate(score_digits):
        digit = int(digit_char)
        texture_id = digit_texture_id_dict.get(digit)
        drawCube(SCORE_POS[0] + i * DIGIT_SPACING, SCORE_POS[1], SCORE_POS[2], (1.0, 1.0, 1.0), texture_id)

# ゴーストブロックを半透明で描画する
def drawGhostCube(x=0.0, y=0.0, z=0.0, color=(1.0, 1.0, 1.0)):
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    light_enabled = glGetBooleanv(GL_LIGHTING)
    cull_enabled = glGetBooleanv(GL_CULL_FACE)

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

# 固定済みブロックを描画する
def drawField():
    for y in range(state.field.shape[0]):
        for x in range(state.field.shape[1]):
            val = state.field[y][x]
            if val != 0:
                drawCube(x, y, 0, ID_TO_COLOR[val])

# 床面を描画する
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

# 背景面を描画する
def drawPlaneZ(size):
    
    glBegin(GL_QUADS)
    glNormal3f(0.0, 0.0, 0.5)
    for i in range(size):
        for j in range(size):
            x = i - 0.5 * size
            y = j - 0.5 * size
            if (i + j) % 2 == 0:
                if state.game_over:
                    glColor3f(0.0, 0.0, 0.0)
                else:    
                    glColor3f(0.0, 0.0, 0.0)
            else:
                glColor3f(0.2, 0.2, 0.2)
                
            glVertex3f(x,     y,     0)
            glVertex3f(x,     y + 1, 0)
            glVertex3f(x + 1, y + 1, 0)
            glVertex3f(x + 1, y,     0)
    glEnd()

# フィールドのグリッドと外枠を描画する
def drawFrame():
    glDisable(GL_LIGHTING)
    z = FIELD_FRAME_Z

    glColor3f(*FIELD_GRID_COLOR)

    glLineWidth(FIELD_GRID_WIDTH)
    glBegin(GL_LINES)
    for x in range(1, FIELD_WIDTH):
        glVertex3f(x, 0, z)
        glVertex3f(x, FIELD_HEIGHT, z)

    for y in range(1, FIELD_HEIGHT):
        glVertex3f(0, y, z)
        glVertex3f(FIELD_WIDTH, y, z)
    glEnd()

    glLineWidth(FIELD_OUTLINE_WIDTH)
    glBegin(GL_LINE_LOOP)
    glVertex3f(0, 0, z)
    glVertex3f(FIELD_WIDTH, 0, z)
    glVertex3f(FIELD_WIDTH, FIELD_HEIGHT, z)
    glVertex3f(0, FIELD_HEIGHT, z)
    glEnd()

    glLineWidth(1.0)
    glEnable(GL_LIGHTING)

# NEXT/HOLD用のプレビュー枠を描画する
def drawPreviewFrame():
    light_enabled = glIsEnabled(GL_LIGHTING)
    glDisable(GL_LIGHTING)

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

# 指定位置のミノが壁・床・固定ブロックと衝突するか判定する
def check_collision(next_pos, next_mino):
    for i in range(4):
        x = int(next_pos[0] + next_mino[i][0])
        y = int(next_pos[1] + next_mino[i][1])

        if x < 0 or x >= FIELD_WIDTH or y < 0:
            return True

        if 0 <= y < state.field.shape[0] and state.field[y][x] != 0:
            return True
            
    return False

# 揃ったラインを消去し、消した行数を返す
def check_line_clear():
    lines_cleared = 0
    new_field = np.zeros((FIELD_HEIGHT, FIELD_WIDTH), dtype=int)
    new_y = 0
    
    for y in range(FIELD_HEIGHT):
        if np.all(state.field[y] != 0):
            lines_cleared += 1
            print(f"Line {y} cleared!")
        else:
            new_field[new_y] = state.field[y]
            new_y += 1
            
    state.field = new_field
    return lines_cleared

# キーボード入力を処理する
def keyboard(window, key, scancode, action, mods):

    if action not in (glfw.PRESS, glfw.REPEAT):
        return

    if action == glfw.PRESS and key == KEY_QUIT:
        glfw.set_window_should_close(window, True)
        print("Quit key pressed - exiting")
        return

    if action == glfw.PRESS and key == KEY_RESTART:
        reset_game()
        return

    if state.game_over:
        return

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

    if not state.drop_switch:
        return

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

    elif key == KEY_SOFT_DROP:
        next_pos = state.mino_pos + np.array([0, -1, 0])
        if not check_collision(next_pos, state.mino):
            state.mino_pos = next_pos
            state.last_drop_time = glfw.get_time()
            state.lock_timer = None

    elif action == glfw.PRESS and key == KEY_HARD_DROP:
        while not check_collision(state.mino_pos + np.array([0, -1, 0]), state.mino):
            state.mino_pos[1] -= 1
        lock_and_spawn_mino()
        state.last_drop_time = glfw.get_time()

    elif action == glfw.PRESS and key in KEY_ROTATE_RIGHT:
        rotate_mino("right")

    elif action == glfw.PRESS and key in KEY_ROTATE_LEFT:
        rotate_mino("left")

    elif action == glfw.PRESS and key in KEY_HOLD:
        if state.can_hold:
            if state.hold_mino_type is None:
                state.hold_mino_type = state.current_mino_type
                state.current_mino_type = state.next_mino_queue.pop(0)
                refill_next_queue()
                state.next_mino_type = state.next_mino_queue[0]
                
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
    
# ウィンドウ再描画コールバック
def refresh(window):
    display()
    glfw.swap_buffers(window)
    
# 射影行列を設定する
def perspective(width, height):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, width / height, 1.0, 100.0)
    glMatrixMode(GL_MODELVIEW)
    
# ウィンドウサイズ変更時に射影行列を更新する
def resize(window, width, height):
    perspective(width, height)
    
# OpenGLとゲーム状態を初期化する
def init():
    glClearColor(0.2, 0.2, 0.2, 1.0)
    perspective(WINDOW_WIDTH, WINDOW_WIDTH)
    
    glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    
    glEnable(GL_COLOR_MATERIAL) 
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
    
    reset_game()
    
# エントリーポイント
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

'''
鬩包ｽｶ驗呻ｽｫ郢晢ｽｻ/ 鬩包ｽｶ驗呻ｽｫ郢晢ｽｻ         |  鬮ｯ譎｢・ｽ・ｾ郢晢ｽｻ繝ｻ・ｦ鬮ｯ・ｷ繝ｻ・ｿ郢晢ｽｻ繝ｻ・ｳ鬯ｩ蜍溪・繝ｻ・ｽ繝ｻ・ｻ鬮ｯ・ｷ鬮ｦ・ｪ郢晢ｽｻ
鬩包ｽｶ驗呻ｽｫ郢晢ｽｻ             |  鬩幢ｽ｢繝ｻ・ｧ郢晢ｽｻ繝ｻ・ｽ鬩幢ｽ｢隴弱・・ｽ・ｼ隴∫浹螟石碑ｭ取得・ｽ・ｳ繝ｻ・ｨ繝ｻ蜿厄ｽｺ・ｽ繝ｻ・ｹ隴擾ｽｴ郢晢ｽｻ驛｢譎｢・ｽ・ｻ
Space          |  鬩幢ｽ｢隴乗・・ｽ・ｸ驗呻ｽｫ郢晢ｽｻ鬩幢ｽ｢隴取得・ｽ・ｳ繝ｻ・ｨ驛｢譎｢・ｽ・ｩ鬩幢ｽ｢隴趣ｽ｢繝ｻ・ｽ繝ｻ・ｭ鬩幢ｽ｢隴擾ｽｴ郢晢ｽｻ驛｢譎｢・ｽ・ｻ
鬩包ｽｶ驗呻ｽｫ郢晢ｽｻ/ E          |  鬮ｯ・ｷ繝ｻ・ｿ郢晢ｽｻ繝ｻ・ｳ鬮ｯ諛・ｻｸ繝ｻ・ｫ郢晢ｽｻ繝ｻ・ｽ繝ｻ・ｻ郢晢ｽｻ繝ｻ・｢
Q / Left Ctrl  |  鬮ｯ譎｢・ｽ・ｾ郢晢ｽｻ繝ｻ・ｦ鬮ｯ諛・ｻｸ繝ｻ・ｫ郢晢ｽｻ繝ｻ・ｽ繝ｻ・ｻ郢晢ｽｻ繝ｻ・｢
C / Left Shift |  鬩幢ｽ｢隴取得・ｽ・ｸ陷ｷ・ｶ郢晢ｽｻ鬩幢ｽ｢隴趣ｽ｢繝ｻ・ｽ繝ｻ・ｫ鬩幢ｽ｢隴擾ｽｴ郢晢ｽｻ
P              |  鬩幢ｽ｢隴弱・・ｺ・｢驛｢譎｢・ｽ・ｻ鬩幢ｽ｢繝ｻ・ｧ郢晢ｽｻ繝ｻ・ｺ/鬮ｯ・ｷ・つ髯憺屮・ｽ・ｼ髯晢ｽｷ郢晢ｽｻ
R              |  鬩幢ｽ｢隴趣ｽ｢繝ｻ・ｽ繝ｻ・ｪ鬩幢ｽ｢繝ｻ・ｧ郢晢ｽｻ繝ｻ・ｹ鬩幢ｽ｢繝ｻ・ｧ郢晢ｽｻ繝ｻ・ｿ鬩幢ｽ｢隴趣ｽ｢繝ｻ・ｽ繝ｻ・ｼ鬩幢ｽ｢隴擾ｽｴ郢晢ｽｻ
Escape         |  鬯ｩ謳ｾ・ｽ・ｨ驛｢・ｧ郢晢ｽｻ繝ｻ・ｽ繝ｻ・ｺ驛｢譎｢・ｽ・ｻ
'''
