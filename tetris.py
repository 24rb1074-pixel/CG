import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np


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
[0, 1], [1, 2], [2, 3], [3, 0], # 底面の辺
[4, 5], [5, 6], [6, 7], [7, 4], # 上面の辺
[0, 4], [1, 5], [2, 6], [3, 7]  # 側面の辺
])

# 時計回りに設定
face = np.array([
[3, 2, 1, 0], # D-C-B-A を結ぶ面
[2, 1, 5, 6], # C-B-F-G を結ぶ面
[6, 5, 4, 7], # G-F-E-H を結ぶ面
[3, 7, 4, 0], # D-H-E-A を結ぶ面
[0, 4, 5, 1], # A-E-F-B を結ぶ面
[7, 3, 2, 6]  # H-D-C-G を結ぶ面
])

normals = np.array([
[ 0.0, 0.0,-1.0], # D-C-B-A を結ぶ面の法線
[ 0.0, 1.0, 0.0], # C-B-F-G を結ぶ面の法線
[ 0.0, 0.0, 1.0], # G-F-E-H を結ぶ面の法線
[ 0.0,-1.0, 0.0], # D-H-E-A を結ぶ面の法線
[-1.0, 0.0, 0.0], # A-E-F-B を結ぶ面の法線
[ 1.0, 0.0, 0.0]  # H-D-C-G を結ぶ面の法線
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

# 色の定義
COLORS = {
    "I": (0.0, 1.0, 1.0),  # シアン
    "O": (1.0, 1.0, 0.0),  # 黄
    "T": (1.0, 0.0, 1.0),  # マゼンタ
    "S": (0.0, 1.0, 0.0),  # 緑
    "Z": (1.0, 0.0, 0.0),  # 赤
    "J": (0.0, 0.0, 1.0),  # 青
    "L": (1.0, 0.5, 0.0)   # オレンジ
}

# フィールドに保存するための色ID
COLOR_IDS = {"I": 1, "O": 2, "T": 3, "S": 4, "Z": 5, "J": 6, "L": 7}
# 描画時にIDから色を取り出すための辞書
ID_TO_COLOR = {1: COLORS["I"], 2: COLORS["O"], 3: COLORS["T"], 4: COLORS["S"], 5: COLORS["Z"], 6: COLORS["J"], 7: COLORS["L"]}

# ==========================
# グローバル変数の宣言
current_mino_type = None
next_mino_type = None
hold_mino_type = None
can_hold = True 
mino_pos = None
mino = None
drop_switch = False
last_drop_time = 0.0
field = None
game_over = False
score = 0
total_lines_cleared = 0
last_total_lines_cleared = 0
gaming_color_param = 1.0  
lock_delay = 0.5  # ミノが着地してから固定されるまでの遅延時間
lock_timer = None  # ロックタイマーの初期化
lock_reset_counter = 0  # ロックタイマーリセットの回数をカウントする変数
max_lock_resets = 15  # ロックタイマーをリセットできる最大回数
mino_bag = []  # ミノのバグを管理するリスト
pause_started_time = None  # ポーズを開始した時刻

# ゲーム状態を初期化（リセット）する関数
def reset_game():
    global current_mino_type, next_mino_type, hold_mino_type, can_hold
    global mino_pos, mino, drop_switch, last_drop_time, field, game_over, score, total_lines_cleared, gaming_color_param, last_total_lines_cleared
    global lock_timer, lock_reset_counter, mino_bag, pause_started_time
    
    score = 0
    total_lines_cleared = 0
    last_total_lines_cleared = 0
    gaming_color_param = 1.0
    lock_reset_counter = 0
    lock_timer = None
    pause_started_time = None

    mino_bag.clear()  # ミノのバグをクリア
    current_mino_type = get_next_mino_type()
    next_mino_type = get_next_mino_type()
    hold_mino_type = None
    can_hold = True
    mino_pos = np.array([4.0, 19.0, 0.0])
    mino = np.copy(minos[current_mino_type])
    
    drop_switch = True
    last_drop_time = glfw.get_time()
    field = np.zeros((20, 10), dtype=int) 
    game_over = False
    print("Game Reset!")
# ==========================


def get_next_mino_type():
    global mino_bag
    
    if not mino_bag:
        mino_bag.extend(list(minos.keys()))
        np.random.shuffle(mino_bag)
    return mino_bag.pop()

# ミノをフィールドに固定し、新しいミノを生成する
def lock_and_spawn_mino():
    global field, mino_pos, mino, current_mino_type, drop_switch, next_mino_type, hold_mino_type
    global can_hold, game_over, score, total_lines_cleared, last_total_lines_cleared, lock_timer, lock_reset_counter

    if game_over:
        return  # ゲームオーバー時は何もしない

    # 固定処理
    color_id = COLOR_IDS[current_mino_type]
    # ミノの4つのブロックをフィールドに固定
    for i in range(4):
        x = int(mino_pos[0] + mino[i][0])
        y = int(mino_pos[1] + mino[i][1])
        if 0 <= x < 10 and 0 <= y < 20 and field[y][x] == 0:
            field[y][x] = color_id

    # 行のクリアをチェック
    lines_cleared = check_line_clear()
    if lines_cleared == 1:
        score += 100
    elif lines_cleared == 2:
        score += 300
    elif lines_cleared == 3:
        score += 500
    elif lines_cleared == 4:
        score += 800 
    
    # トータルライン数を更新
    total_lines_cleared += lines_cleared
    
    # レベルアップの判定（10ラインごとにレベルアップ）
    if total_lines_cleared // 10 > last_total_lines_cleared // 10:
        print(f"Level Up! Total Lines Cleared: {total_lines_cleared}")
    
    # 最後のレベルアップ時のライン数を更新
    last_total_lines_cleared = total_lines_cleared

    # 新しいミノを生成
    mino_pos = np.array([4.0, 19.0, 0.0])
    current_mino_type = next_mino_type
    next_mino_type = get_next_mino_type()
    mino = np.copy(minos[current_mino_type])
    can_hold = True  # 新しいミノが生成されたのでホールド可能にする
    lock_timer = None  # 新しいミノが生成されたのでロックタイマーをリセット
    lock_reset_counter = 0  # ロックリセットカウンターをリセット


    # ゲームオーバー判定
    if check_collision(mino_pos, mino):
        print("Game Over!")
        print(f"Final Score: {score}, Total Lines Cleared: {total_lines_cleared}")
        print("Press R to restart the game.")
        drop_switch = False
        game_over = True

def rotate_mino(direction):
    global mino, mino_pos, lock_timer, lock_reset_counter, current_mino_type
    
    if current_mino_type == "O":
        return  # Oミノは回転しない
    
    new_mino = np.copy(mino)
    if direction == "left":
        for i in range(4):
            new_mino[i][0] = -mino[i][1]
            new_mino[i][1] = mino[i][0]
    elif direction == "right":
        for i in range(4):
            new_mino[i][0] = mino[i][1]
            new_mino[i][1] = -mino[i][0]
    else:
        return  # 無効な方向の場合は何もしない
            
    rotated = False

    # 回転後の位置で衝突判定
    # キックテストを試す
    for test in kick_tests:
        test_pos = mino_pos + np.array(test)
        if not check_collision(test_pos, new_mino):
            mino_pos = test_pos
            mino = new_mino
            rotated = True
            break  # キック成功したらループを抜ける
    
    if rotated and lock_reset_counter < max_lock_resets and lock_timer is not None:
        lock_timer = None  # 回転したのでロックタイマーをリセット
        lock_reset_counter += 1  # リセット回数をカウント


def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # モデルビュー変換行列の設定
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    gluLookAt(15.0, 10.0, 30.0, 5.0, 10.0, 0.0, 0.0, 1.0, 0.0)
    
    glFrontFace(GL_CW) # 時計回りを表面とする
    glCullFace(GL_BACK)
    
    light_position = [5.0, 10.0, 20.0, 1.0]
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    
    global drop_switch, mino_pos, time, minos, mino, field, last_drop_time, current_mino_type, next_mino_type, hold_mino_type, can_hold, game_over, score, total_lines_cleared, last_total_lines_cleared, lock_timer
    current_time = glfw.get_time()   
    
    # === ロジック処理 ===
    if not game_over and drop_switch:
        fall_interval = max(
            0.1,
            1.0 - (total_lines_cleared // 10) * 0.1
        )

        # 通常落下
        if current_time - last_drop_time > fall_interval:
            last_drop_time = current_time
            next_pos = mino_pos + np.array([0, -1, 0])

            if not check_collision(next_pos, mino):
                mino_pos = next_pos

        # 接地状態を毎フレーム確認
        grounded = check_collision(
            mino_pos + np.array([0, -1, 0]),
            mino
        )

        if grounded:
            # 接地した瞬間に一度だけ開始
            if lock_timer is None:
                lock_timer = current_time

            # 0.5秒経過したら固定
            elif current_time - lock_timer >= lock_delay:
                lock_and_spawn_mino()
        else:
            # 接地状態から抜けた
            lock_timer = None



        # === 描画処理 ===
    
    # 1. 固定されたフィールドを描画    
    glPushMatrix()
    drawField() 
    drawFrame()
    glPopMatrix()
    
    # 2. ネクストとホールドを描画
    glPushMatrix()
    glTranslatef(13.0, 15.0, 0.0)
    drawPreviewFrame()
    for i in range(4):
        drawCube(minos[next_mino_type][i][0], minos[next_mino_type][i][1], minos[next_mino_type][i][2], COLORS[next_mino_type])
    glPopMatrix()

    glPushMatrix()
    glTranslatef(-4.0, 15.0, 0.0)
    drawPreviewFrame()
    if hold_mino_type is not None:
        for i in range(4):
            drawCube(minos[hold_mino_type][i][0], minos[hold_mino_type][i][1], minos[hold_mino_type][i][2], COLORS[hold_mino_type])
    glPopMatrix()
        
    # 3. 操作中のミノとゴーストを描画（※ゲームオーバー時は描画しない）
    if not game_over:
        # ゴーストミノ
        ghost_pos = mino_pos.copy()
        # ゴーストミノの位置を計算する
        while not check_collision(ghost_pos + np.array([0, -1, 0]), mino):
            ghost_pos += np.array([0, -1, 0])

        for i in range(4):
            drawGhostCube(ghost_pos[0] + mino[i][0], ghost_pos[1] + mino[i][1], ghost_pos[2] + mino[i][2], COLORS[current_mino_type])

        # 操作中ミノ
        glPushMatrix()
        glTranslatef(mino_pos[0], mino_pos[1], mino_pos[2]) 
        for i in range(4):
            drawCube(mino[i][0], mino[i][1], mino[i][2], COLORS[current_mino_type]) 
        glPopMatrix()
        
    # 背景の装飾
    # 床
    glPushMatrix()
    glTranslatef(5.0, 0.0, -1.0)
    drawPlaneY(12)
    glPopMatrix()

    # 背景
    glPushMatrix()
    glTranslatef(5.0, 10.0, -1.0)
    drawPlaneZ(22)
    glPopMatrix()
        
    
# ブロックを描画する関数
def drawCube(x=0.0, y=0.0, z=0.0, color=(1.0, 1.0, 1.0)):
    light_enabled = glGetBooleanv(GL_LIGHTING)

    glPushMatrix()
    glTranslatef(x, y, z)

    # 面を描画
    glEnable(GL_LIGHTING)

    mat_specular = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular)
    glMaterialf(GL_FRONT, GL_SHININESS, 128.0)

    for i in range(6):
        if game_over:
            glColor3fv((0.5, 0.5, 0.5))
        else:
            glColor3fv(color)

        glBegin(GL_QUADS)
        for j in range(4):
            glNormal3fv(normals[i])
            glVertex3fv(vertex[face[i][j]])
        glEnd()

    # 輪郭線を描画
    glDisable(GL_LIGHTING)
    glColor3f(0.0, 0.0, 0.0)
    glLineWidth(2.0)

    glBegin(GL_LINES)
    for i in range(12):
        glVertex3fv(vertex[edge[i][0]])
        glVertex3fv(vertex[edge[i][1]])
    glEnd()

    glPopMatrix()

    # 元のライティング状態に戻す
    if light_enabled:
        glEnable(GL_LIGHTING)
    else:
        glDisable(GL_LIGHTING)

# ゴーストブロックを描画する関数
def drawGhostCube(x=0.0, y=0.0, z=0.0, color=(1.0, 1.0, 1.0)):
    # ゴーストブロックは半透明で描画する
    glEnable(GL_BLEND) # ブレンドを有効にする
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) # ブレンド関数を設定

    light_enabled = glGetBooleanv(GL_LIGHTING)
    glEnable(GL_LIGHTING)
    glPushMatrix()
    glTranslatef(x, y, z)
    
    matWhite = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    glMaterialfv(GL_FRONT, GL_SPECULAR, matWhite)
    glMaterialf(GL_FRONT, GL_SHININESS, 32.0)
    
    for i in range(6):
        glColor4f(color[0], color[1], color[2], 0.3) # 半透明の色を設定
        glBegin(GL_QUADS)
        for j in range(4):
            glNormal3fv(normals[i])
            glVertex3fv(vertex[face[i][j]])
        glEnd()
    glPopMatrix()

    glDepthMask(GL_TRUE) # 後続の通常描画では深度書き込みを元に戻す
    glDisable(GL_BLEND) # ブレンドを無効にする
    
    if not light_enabled:
        glDisable(GL_LIGHTING)

# フィールドを描画する関数
def drawField():
    for y in range(field.shape[0]):
        for x in range(field.shape[1]):
            val = field[y][x]
            if val != 0:
                drawCube(x, y, 0, ID_TO_COLOR[val])

# Y軸方向の平面を描画する関数
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

# Z軸方向の平面を描画する関数
def drawPlaneZ(size):
    global game_over
    
    glBegin(GL_QUADS)
    glNormal3f(0.0, 0.0, 0.5)
    for i in range(size):
        for j in range(size):
            x = i - 0.5 * size
            y = j - 0.5 * size
            if (i + j) % 2 == 0:
                if game_over:
                    glColor3f(0.0, 0.0, 0.0)  # 灰色
                else:    
                    glColor3f(0.0, 0.0, 0.0)  # 明るい灰色
            else:
                glColor3f(0.2, 0.2, 0.2)
                
            glVertex3f(x,     y,     0)
            glVertex3f(x,     y + 1, 0)
            glVertex3f(x + 1, y + 1, 0)
            glVertex3f(x + 1, y,     0)
    glEnd()

# フレームを描画する関数
def drawFrame():
    glDisable(GL_LIGHTING) # ライティングを無効にして確実に色を出す
    z = 1.01 # ブロックの前面(1.0)よりほんの少し手前に描く

    glColor3f(0.7, 0.7, 0.7)  # フレームの色を設定

    # 内側のグリッドを通常の太さで描画
    glLineWidth(2.0)
    glBegin(GL_LINES)
    for x in range(1, 10):
        glVertex3f(x, 0, z)
        glVertex3f(x, 20, z)

    for y in range(1, 20):
        glVertex3f(0, y, z)
        glVertex3f(10, y, z)
    glEnd()

    # 外枠だけを太く描画
    glLineWidth(4.0)
    glBegin(GL_LINE_LOOP)
    glVertex3f(0, 0, z)
    glVertex3f(10, 0, z)
    glVertex3f(10, 20, z)
    glVertex3f(0, 20, z)
    glEnd()

    # 後続の描画へ線幅を残さない
    glLineWidth(1.0)
    glEnable(GL_LIGHTING) # ライティングを元に戻す


# NEXT/HOLDプレビューの外枠を描画する関数
def drawPreviewFrame():
    light_enabled = glIsEnabled(GL_LIGHTING)
    glDisable(GL_LIGHTING)

    # 全種類のミノが収まる6x4の枠
    left, right = -2.0, 4.0
    bottom, top = -1.0, 3.0
    z = 1.01

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

# 衝突判定関数
def check_collision(next_pos, next_mino):
    # 4つのブロックそれぞれについて調べる
    for i in range(4):
        # 1. フィールドの絶対座標を計算する（基準位置 ＋ 相対位置）
        x = int(next_pos[0] + next_mino[i][0])
        y = int(next_pos[1] + next_mino[i][1])

        # 2. 壁や床を突き抜けていないかチェック
        if x < 0 or x >= 10 or y < 0:
            return True # 衝突した！

        # 3. すでにフィールドに固定されているブロックと重なっていないかチェック
        if 0 <= y < field.shape[0] and field[y][x] != 0:
            return True # 衝突した！
            
    return False # どこにも衝突しなかった（移動OK！）

# ライン消去をチェックする関数
def check_line_clear():
    global field
    lines_cleared = 0
    # ライン消去アルゴリズム
    new_field = np.zeros((20, 10), dtype=int)
    new_y = 0
    
    for y in range(20):
        if np.all(field[y] != 0): # その行が全て埋まっているか
            lines_cleared += 1
            print(f"Line {y} cleared!")
        else:
            # 埋まっていない行だけを新しいフィールドに詰めていく
            new_field[new_y] = field[y]
            new_y += 1
            
    field = new_field # フィールドを上書き
    return lines_cleared

# キーボード入力のコールバックを登録
def keyboard(window, key, scancode, action, mods):
    global drop_switch, mino_pos, mino, current_mino_type, last_drop_time
    global next_mino_type, hold_mino_type, can_hold, game_over
    global lock_timer, lock_reset_counter, pause_started_time, score, total_lines_cleared

    # RELEASEイベントは使用しない
    if action not in (glfw.PRESS, glfw.REPEAT):
        return

    # 終了とリスタートは常に受け付ける
    if action == glfw.PRESS and key == glfw.KEY_Q:
        glfw.set_window_should_close(window, True)
        print("Q key pressed - exiting")
        return

    if action == glfw.PRESS and key == glfw.KEY_R:
        reset_game()
        return

    if game_over:
        return

    # Escでポーズ／再開。停止中の時間を各タイマーから除外する
    if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
        current_time = glfw.get_time()
        if drop_switch:
            drop_switch = False
            pause_started_time = current_time
            print("Paused")
        else:
            if pause_started_time is not None:
                paused_duration = current_time - pause_started_time
                last_drop_time += paused_duration
                if lock_timer is not None:
                    lock_timer += paused_duration
            drop_switch = True
            pause_started_time = None
            print("Resumed")
        refresh(window)
        return

    # ポーズ中はゲーム操作を受け付けない
    if not drop_switch:
        return

    # 左右移動（押しっぱなし対応）
    if key == glfw.KEY_RIGHT:
        if not check_collision(mino_pos + np.array([1, 0, 0]), mino):
            mino_pos[0] += 1
            if lock_reset_counter < max_lock_resets and lock_timer is not None:
                lock_timer = None
                lock_reset_counter += 1

    elif key == glfw.KEY_LEFT:
        if not check_collision(mino_pos + np.array([-1, 0, 0]), mino):
            mino_pos[0] -= 1
            if lock_reset_counter < max_lock_resets and lock_timer is not None:
                lock_timer = None
                lock_reset_counter += 1

    # ↓：ソフトドロップ（押しっぱなし対応）
    elif key == glfw.KEY_DOWN:
        next_pos = mino_pos + np.array([0, -1, 0])
        if not check_collision(next_pos, mino):
            mino_pos = next_pos
            last_drop_time = glfw.get_time()
            lock_timer = None

    # Space：ハードドロップ
    elif action == glfw.PRESS and key == glfw.KEY_SPACE:
        while not check_collision(mino_pos + np.array([0, -1, 0]), mino):
            mino_pos[1] -= 1
        lock_and_spawn_mino()
        last_drop_time = glfw.get_time()

    # ↑／X：右回転
    elif action == glfw.PRESS and key in (glfw.KEY_UP, glfw.KEY_X):
        rotate_mino("right")

    # Z／Left Ctrl：左回転
    elif action == glfw.PRESS and key in (glfw.KEY_Z, glfw.KEY_LEFT_CONTROL):
        rotate_mino("left")

    # C／Left Shift：ホールド
    elif action == glfw.PRESS and key in (glfw.KEY_C, glfw.KEY_LEFT_SHIFT):
        if can_hold:
            if hold_mino_type is None:
                hold_mino_type = current_mino_type
                current_mino_type = next_mino_type
                next_mino_type = get_next_mino_type()
            else:
                current_mino_type, hold_mino_type = hold_mino_type, current_mino_type

            mino = np.copy(minos[current_mino_type])
            mino_pos = np.array([4.0, 19.0, 0.0])
            
            if check_collision(mino_pos, mino):
                print("Game Over!")
                print(f"Final Score: {score}, Total Lines Cleared: {total_lines_cleared}")
                print("Press R to restart the game.")
                drop_switch = False
                game_over = True
                return
            
            lock_timer = None
            lock_reset_counter = 0
            last_drop_time = glfw.get_time()
            can_hold = False
            
            

    refresh(window)
    
def refresh(window):
    display()
    glfw.swap_buffers(window)
    
def perspective(width, height):
    # 透視変換行列の設定
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, width / height, 1.0, 100.0)
    # モデルビュー変換行列の設定
    glMatrixMode(GL_MODELVIEW)
    
def resize(window, width, height):
    perspective(width, height)
    
def init():
    glClearColor(0.2, 0.2, 0.2, 1.0)
    perspective(512, 512)
    
    glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    
    glEnable(GL_COLOR_MATERIAL) 
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
    
    # ゲーム状態を初期化
    reset_game()
    
def main():
    glfw.init()
    
    glfw.window_hint(glfw.SAMPLES, 4)
    
    window = glfw.create_window(512, 768, "3D Tetris", None, None)
    glfw.make_context_current(window)
    init()
    
    glfw.set_window_refresh_callback(window, refresh)
    glfw.set_key_callback(window, keyboard)
    
    refresh(window)
    
    while not glfw.window_should_close(window):
        display()
        glfw.swap_buffers(window)
        glfw.poll_events()
    glfw.destroy_window(window)
    glfw.terminate()

if __name__ == "__main__":
    main()

#==========================
# キー設定一覧
'''
← / →          |  左右移動
↓              |  ソフトドロップ
Space          |  ハードドロップ
↑ / X          |  右回転
Z / Left Ctrl  |  左回転
C / Left Shift |  ホールド
Esc            |  ポーズ/再開
R              |  リスタート
Q              |  終了
'''
#==========================
