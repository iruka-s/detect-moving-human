import cv2
import numpy as np
 
def make_diff_frame(height, width, previous_frame, next_frame):
    result_frame = np.copy(next_frame)
    result_frame[np.all(np.isclose(previous_frame, next_frame, rtol=0, atol=3), axis=2)] = np.array([255, 255, 255], dtype='uint8')
    return result_frame



# 動画読み込みの設定
movie = cv2.VideoCapture('mov2.mp4')
 
# 動画ファイル保存用の設定
fps = int(movie.get(cv2.CAP_PROP_FPS))
width = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter('video_out2.mp4', fourcc, fps, (width, height), True)
all_frame_num = movie.get(cv2.CAP_PROP_FRAME_COUNT)

# 動画と同じサイズの0埋め配列
previous_frame = np.array([[[0, 0, 0]] * width] * height, dtype='uint8')

# ファイルからフレームを1枚ずつ取得して動画処理後に保存する
progress_num = 1
while True:
    ret, frame = movie.read()

    # フレームが取得できない場合はループを抜ける
    if not ret:
        break
    
    # 一つ前のフレームとの差分を取る
    print("now processing:", end=" ")
    print(progress_num, end=" / ")
    print(all_frame_num)
    result_frame = make_diff_frame(height, width, previous_frame, frame)
    previous_frame = np.copy(frame)

    video.write(result_frame)
    progress_num = progress_num + 1
    
 
# 撮影用オブジェクトとウィンドウの解放
movie.release()