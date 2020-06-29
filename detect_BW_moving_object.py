import cv2

# 動画読み込みの設定
movie = cv2.VideoCapture('mov2.mp4')
 
# 動画ファイル保存用の設定
fps = int(movie.get(cv2.CAP_PROP_FPS))
width = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter('video_out2.mp4', fourcc, fps, (width, height), True)
all_frame_num = movie.get(cv2.CAP_PROP_FRAME_COUNT)

# 背景差分の設定
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

# ファイルからフレームを1枚ずつ取得して動画処理後に保存する
while True:
    ret, frame = movie.read()

    # フレームが取得できない場合はループを抜ける
    if not ret:
        break
    
    # 一つ前のフレームとの差分を取る
    fgmask = fgbg.apply(frame)

    video.write(previous_frame)
    
 
# 撮影用オブジェクトとウィンドウの解放
movie.release()