import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

def drawAxis():
    glBegin(GL_LINES)
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, 0.0); glVertex3f(2.0, 0.0, 0.0)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(0.0, 0.0, 0.0); glVertex3f(0.0, 2.0, 0.0)
    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(0.0, 0.0, 0.0); glVertex3f(0.0, 0.0, 2.0)
    glEnd()

def from_spherical(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def drawSphere(radius):
    glBegin(GL_QUADS)
    for i in range(8):
        for j in range(16):
            theta = i / 8.0 * np.pi
            phi = j / 16.0 * 2.0 * np.pi
            dtheta = 1.0 / 8.0 * np.pi
            dphi = 1.0 / 16.0 * 2.0 * np.pi
            glVertex3fv(from_spherical(radius, theta, phi))
            glVertex3fv(from_spherical(radius, theta + dtheta, phi))
            glVertex3fv(from_spherical(radius, theta + dtheta, phi + dphi))
            glVertex3fv(from_spherical(radius, theta, phi + dphi))
    glEnd()

def drawWireSphere(radius):
    for i in range(8):
        for j in range(16):
            theta = i / 8.0 * np.pi
            phi = j / 16.0 * 2.0 * np.pi
            dtheta = 1.0 / 8.0 * np.pi
            dphi = 1.0 / 16.0 * 2.0 * np.pi
            glBegin(GL_LINE_LOOP)
            glVertex3fv(from_spherical(radius, theta, phi))
            glVertex3fv(from_spherical(radius, theta + dtheta, phi))
            glVertex3fv(from_spherical(radius, theta + dtheta, phi + dphi))
            glVertex3fv(from_spherical(radius, theta, phi + dphi))
            glEnd()

def drawPlaneY(size):
    for i in range(int(size)):
        for j in range(int(size)):
            x = i - 0.5 * size
            z = j - 0.5 * size
            if (i + j) % 2 == 0:
                glColor3f(1, 1, 1)
            else:
                glColor3f(0.2, 0.2, 0.2)
            
            glBegin(GL_QUADS)
            glVertex3f(x, 0, z)
            glVertex3f(x + 1, 0, z)
            glVertex3f(x + 1, 0, z + 1)
            glVertex3f(x, 0, z + 1)
            glEnd()

# === 課題内容 ===

# === グローバル変数（初期値） ===
theta = 0.0          # 角度（ラジアン）
omega = 0.0          # 初期の角速度（ラジアン/秒）
alpha = 1.5          # 角加速度（1秒間に角速度がどれだけ増えるか）
radius = 5.0         # 初期の半径
radius_speed = 0.5   # 半径が縮むスピード（1秒間にどれだけ縮むか）

pos = np.array([radius * np.cos(theta), 5.0, radius * np.sin(theta)], dtype=float)
# X軸とZ軸の速度は極座標で直接位置を計算するため不要になり、Y軸の速度のみ使用します
vel = np.array([0.0, 0.0, 0.0], dtype=float) 
g = np.array([0.0, -9.8, 0.0], dtype=float)

trajectory = [] # 軌跡を保存するリスト

def display():
    global pos, vel, omega, theta, radius, alpha
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(8.0, 10.0, 12.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    
    dt = 0.01
    
    # === 物理演算の更新 ===
    
    # -- A. 極座標（XZ平面）の更新 --
    if radius > 0.0:
        radius -= radius_speed * dt
        if radius < 0.0:
            radius = 0.0 # 中心に到達したら半径を0で固定
            omega = 0.0  # 回転も止める
            alpha = 0.0  # 角加速度も0にする
            theta = 0.0  # 角度も0にする
    
    # 半径が0でない限り、角速度を更新して回転させる
    if radius > 0.0: 
        omega += alpha * dt
        
    # 角度を更新
    theta += omega * dt
    
    # 極座標からデカルト座標への変換
    pos[0] = radius * np.cos(theta)
    pos[2] = radius * np.sin(theta)

    
    # -- B. Y軸（自由落下・バウンド）の更新 --
    vel[1] = vel[1] + g[1] * dt
    pos[1] = pos[1] + vel[1] * dt


    # -- C. 床との衝突判定 --
    if pos[1] < 0.5:
        pos[1] = 0.5
        if abs(vel[1]) < 0.5:
            vel[1] = 0.0
        else:
            vel[1] = -vel[1] * 0.9
    
    # 軌跡の保存
    trajectory.append(pos.copy())
    if len(trajectory) > 2000:
        trajectory.pop(0)
            
    # === 描画処理 ===
    # 1. 軌跡の描画 
    glColor3f(1.5, 1.0, 0.0) # 軌跡の色
    glLineWidth(3.0) # 線の太さ
    glBegin(GL_LINE_STRIP)
    for p in trajectory:
        glVertex3fv(p)
    glEnd()
    glLineWidth(1.0)
    
    # 2. ボールの描画
    glPushMatrix()
    glTranslatef(pos[0], pos[1], pos[2])
    glColor3f(0.8, 0.8, 0.8)
    drawSphere(0.5)
    glColor3f(1, 1, 1)
    drawWireSphere(0.5)
    glPopMatrix()
    
    # 3. 背景・軸の描画
    drawPlaneY(10)
    drawAxis()
    
# === ここまで課題内容 ===

def perspective(width, height):
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(30.0, width / height, 1.0, 100.0)
    glMatrixMode(GL_MODELVIEW)

def init():
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.5, 0.5, 0.5, 1.0)
    perspective(512, 512)

def main():
    if not glfw.init():
        return
    glfw.window_hint(glfw.SAMPLES, 4)
    window = glfw.create_window(512, 512, "Trajectory Simulation", None, None)
    if not window:
        glfw.terminate()
        return
        
    glfw.make_context_current(window)
    init()
    
    # サイズ変更時のコールバックをシンプルに登録
    glfw.set_framebuffer_size_callback(window, lambda w, width, height: perspective(width, height))
    
    while not glfw.window_should_close(window):
        display()
        glfw.swap_buffers(window)
        glfw.poll_events()
        
    glfw.destroy_window(window)
    glfw.terminate()

if __name__ == "__main__":
    main()