import argparse
import logging
import time

import cv2
import numpy as np
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


def make_diff_frame(height, width, previous_frame, next_frame):
    result_frame = np.copy(next_frame)
    result_frame[np.all(np.isclose(previous_frame, next_frame, rtol=0, atol=3), axis=2)] = np.array([255, 255, 255], dtype='uint8')
    return result_frame

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))

    movie = cv2.VideoCapture(args.video)

    # 動画ファイル保存用の設定
    fps = int(movie.get(cv2.CAP_PROP_FPS))
    width = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter('video_out.mp4', fourcc, fps, (width, height), True)
    all_frame_num = movie.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # width, height = model_wh(args.resize)

    # 動画と同じサイズの0埋め配列
    previous_frame = np.array([[[0, 0, 0]] * width] * height, dtype='uint8')

    if width > 0 and height > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(width, height))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    

    if movie.isOpened() is False:
        print("Error opening video stream or file")

    progress_num = 1
    while True:
        ret_val, frame = movie.read()

        # フレームが取得できない場合はループを抜ける
        if not ret_val:
            break

        # 進捗表示
        print("now processing:", end=" ")
        print(progress_num, end=" / ")
        print(all_frame_num)
        logger.debug('frame process+')

        # 一つ前のフレームとの差分を取る
        diff_frame = make_diff_frame(height, width, previous_frame, frame)
        previous_frame = np.copy(frame)

        # フレーム内の骨格を検出
        humans = e.inference(diff_frame, resize_to_default=(width > 0 and height > 0), upsample_size=args.resize_out_ratio)
        
        if not args.showBG:
            diff_frame = np.zeros(diff_frame.shape)
        
        logger.debug('postprocess+')

        # 元フレームに骨格を描画
        result_frame = TfPoseEstimator.draw_humans(diff_frame, humans, imgcopy=False)

        logger.debug('show+')
        cv2.putText(result_frame, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        video.write(result_frame)
        progress_num = progress_num + 1
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
logger.debug('finished+')