import cv2 as cv
import numpy as np

def select_img_from_video(video_file):
    # Open a video
    video = cv.VideoCapture(video_file)
    img_select = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        img_select.append(frame)
    video.release()
    return img_select
    

def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    # 이미지 수가 10개 이상이면 등간격으로 10개 선택
    if len(images) > 10:
        interval = len(images) // 10
        selected_images = [images[i] for i in range(0, len(images), interval)]
        selected_images = selected_images[:10]
    else:
        selected_images = images
    
    # Find 2D corner points from selected images
    img_points = []
    for img in selected_images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)
    assert len(img_points) > 0, 'There is no set of complete chessboard points!'
    # Prepare 3D points of the chess board
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points) # Must be `np.float32`
    # Calibrate the camera
    print('Start calibration...')
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)
    
    # Print calibration results
    print("\nCalibration Results:")
    print(f"RMS reprojection error: {ret}")
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)
    print("Rotation vectors:\n", rvecs)
    print("Translation vectors:\n", tvecs)

    return ret, mtx, dist, rvecs, tvecs

def undistort_video(video_file, camera_matrix, dist_coeffs, output_file="output.avi"):
    """
    비디오의 각 프레임을 왜곡 보정합니다.
    
    Args:
        video_file (str): 보정할 원본 비디오 파일
        camera_matrix (np.ndarray): 카메라 내부 매트릭스
        dist_coeffs (np.ndarray): 왜곡 계수
        output_file (str): 보정된 비디오 저장 경로
    """
    # 비디오 읽기
    cap = cv.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}")
        return None

    # 비디오 속성 가져오기
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    
    # 보정된 비디오 저장을 위한 VideoWriter 설정
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # 보정 맵 계산 (한번 계산하면 모든 프레임에 재사용 가능)
    mapx, mapy = cv.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (width, height), cv.CV_32FC1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 보정 맵을 사용하여 왜곡 보정
        dst = cv.remap(frame, mapx, mapy, cv.INTER_LINEAR)
        
        # 결과 저장
        out.write(dst)
        
        # 결과 보여주기 (옵션)
        cv.imshow('Undistorted', dst)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv.destroyAllWindows()
    print(f"Undistorted video saved to {output_file}")
    return output_file

if __name__ == "__main__":
    board_pattern = (10, 7)
    board_cellsize = 25.0
    input_video = 'chess.avi'
    
    # 카메라 캘리브레이션 수행
    images = select_img_from_video(input_video)
    ret, mtx, dist, rvecs, tvecs = calib_camera_from_chessboard(images, board_pattern, board_cellsize)
    
    # 왜곡 보정 수행
    undistort_video(input_video, mtx, dist, "undistorted_chess.avi")
